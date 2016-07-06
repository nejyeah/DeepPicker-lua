-----------------------------------------------------------
-- This script implements a particle pick procedure 
-------------------------------------------------------------
require 'optim'
require 'paths'

autoPicker={}
-- initialize some parameters
function autoPicker:init(mrc_file_autopick, deepModel, scale_size, bin_scale, opt)
	-- mrc_file_autopick: a table, if there are three mrc files, the strcuture is as follows
        --                    { {mrc_file1.mrc},{mrc_file2.mrc},{mrc_file3.mrc}}  or
        --                    { {mrc_file1.mrc,file1_coordinate},{mrc_file2.mrc,file2_coordinate},{mrc_file3.mrc,file3_coordinate}} 
	self.mrc_file_autopick = mrc_file_autopick
	self.deepModel = deepModel
	self.scale_size = scale_size
	self.opt = opt
		
	-- preprocess
	--define the gaussian lowpass filter
	local kernel_size = opt.gaussianKernelSize
	if math.fmod(kernel_size,2) == 0 then kernel_size = kernel_size+1 end 
	local gaussian_kernel = image.gaussian(kernel_size,opt.gaussianSigma,1,true)
	self.gaussian_kernel=gaussian_kernel

	--scale the input mrc to reduce
	local pixel = readmrc(mrc_file_autopick[1])
	local col = pixel:size(2)
	local row = pixel:size(3)
        -- need to be manually set, default is 3
        
        local scale = 1
        if opt.bin then scale = bin_scale end
	--define the scale model
	local scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
	scale_model.weight:fill(1)
	scale_model.bias:fill(0)

	self.scale=scale			-- initialize scale size
	self.scale_model=scale_model		-- initialize scale model
	self.particle_size=math.ceil(opt.particle_size/scale) -- initialize particle size
	self.threshold = opt.threshold		-- initialize the autopick threshold

	self.pick_number = opt.pick_number	
	if self.pick_number>#mrc_file_autopick then self.pick_number=#mrc_file_autopick end
end

-- do parallel autopicking
function autoPicker:parallel_autopick(interation,symbolFinal)
	if not iteration then iteration = 1 end
	local symbolFinal = symbolFinal or true
	self.process_number = self.opt.process
	local pick_parameters = {}
	pick_parameters.opt = self.opt
	pick_parameters.particle_size = self.particle_size
	pick_parameters.scale = self.scale
	pick_parameters.gaussian_kernel = self.gaussian_kernel
	pick_parameters.scale_model = self.scale_model
	pick_parameters.deepModel = self.deepModel
	pick_parameters.mrc_file_autopick = self.mrc_file_autopick
	pick_parameters.process_number = self.process_number
	pick_parameters.scale_size = self.scale_size
	pick_parameters.pick_number = self.pick_number
	
        print("start to pick particle automatically")
	local time_start = sys.clock()
	-- fork 2 process
        parallel.nfork(self.process_number)
        -- exec autopick_worker in each process
        parallel.children:exec(self.autopick_worker)
        -- send the autopick parameters 
        parallel.children:send(pick_parameters)

        local autopick_coordinate_file={}

        for i = 1, self.process_number do
                local results = parallel.children[i]:receive()
                print("number of autopick files in process "..i.. ' :'..#results)
		for j = 1, #results do
                	table.insert(autopick_coordinate_file,results[j])
		end
        end

        local time_end = sys.clock()
        local total_time = (time_end-time_start)/60
        local time_average = total_time/self.pick_number
      	print('total time cost:'..total_time..' min')
      	print('time cost per mrc:'..time_average..' min')

	-- process the autopick_coordinate_file
        -- save the coordinates autopicked in each mrc file
	table.sort(autopick_coordinate_file,function(a,b) return a[1][1][4]<b[1][1][4] end)
	local autopick_coordinate_file_autopick = {}
        local autopick_coordinate_file_cannyEdge = {}
        local autopick_coordinate_file_postProcess= {}
        local autopick_coordinate_file_IceClassifier = {}
	for i=1,#autopick_coordinate_file do
               	table.insert(autopick_coordinate_file_autopick,autopick_coordinate_file[i][1])
               	table.insert(autopick_coordinate_file_cannyEdge,autopick_coordinate_file[i][2])
                table.insert(autopick_coordinate_file_postProcess,autopick_coordinate_file[i][3])
                table.insert(autopick_coordinate_file_IceClassifier,autopick_coordinate_file[i][4])
	end
	-- save the results
        torch.save(paths.concat(self.opt.debugDir,"coordinate_autopick.t7"),autopick_coordinate_file_autopick)
	
	local autopick_file_final = {}
	for i=1,#autopick_coordinate_file_autopick do
		local file_coordinate = {}
		for j=1,#autopick_coordinate_file_autopick[i] do
			if autopick_coordinate_file_autopick[i][j][3] > self.opt.threshold then
				table.insert(file_coordinate,autopick_coordinate_file_autopick[i][j])
			end
		end
		table.insert(autopick_file_final,file_coordinate)
	end
        -- write the coordinate file
	self:write_coordinate(autopick_file_final,self.opt.debugDir,self.opt.coorType)
	-- display the autopick results and save in jpg	
	self:display(autopick_file_final,"autopick_threshold")
	self:display(autopick_coordinate_file_autopick,"autopick_all")
	
	-- process the autopick_coordinate_cannyEdge
	if self.opt.EdgeDetect then
        	torch.save(paths.concat(self.opt.debugDir,"coordinate_Edge.t7"),autopick_coordinate_file_cannyEdge)
		self:display(autopick_coordinate_file_cannyEdge,"cannyEdge")
	end

	-- process the autopick_coordinate_postProcess
	if self.opt.postProcess then
        	torch.save(paths.concat(self.opt.debugDir,"coordinate_postProcess.t7"),autopick_coordinate_file_postProcess)
		self:display(autopick_coordinate_file_postProcess,"postProcess")
	end

	-- process the autopick_coordinate_IceClassifier 
	if self.opt.IceClassifier ~= 'none' then
        	torch.save(paths.concat(self.opt.debugDir,"coordinate_IceClassifier.t7"),autopick_coordinate_file_IceClassifier)
		self:display(autopick_coordinate_file_IceClassifier,"IceClassifier")
	end
	return autopick_coordinate_file_autopick
end

-- each new process forked will run this function
function autoPicker:autopick_worker()
        -- reload libararies
        require 'image'
        require 'sys'
        require 'paths'
        require 'optim'
        require 'nn'
        require '1_datafunctions'
        require 'autoPicker'
	require '1_datafunctions_preprocess'

        -- recieve some parameters from parent
        local pick_parameters = parallel.parent:receive()  -- recieve the parameters

        local opt = pick_parameters.opt                                 -- opt
        if opt.type == 'cuda' then
                require 'cunn'
                torch.setdefaulttensortype('torch.FloatTensor')
                cutorch.setDevice(opt.gpuid)
        elseif opt.type == 'float' then
                torch.setdefaulttensortype('torch.FloatTensor')
	elseif opt.type == 'double' then
                torch.setdefaulttensortype('torch.DoubleTensor')
	else
		error("wrong type:",opt.type)
	end

        local particle_size = pick_parameters.particle_size             -- particle size
        local scale = pick_parameters.scale                             -- scale 
        local gaussian_kernel = pick_parameters.gaussian_kernel         -- gaussian kernel      
        local scale_model = pick_parameters.scale_model                 -- scale model
        local deepModel = pick_parameters.deepModel                     -- CNN model
        local mrc_file_autopick = pick_parameters.mrc_file_autopick     -- mrc file names
        local process_number=pick_parameters.process_number             -- number of total process
        local scale_size=pick_parameters.scale_size                     -- scale size 
        local pick_number = pick_parameters.pick_number			-- number of autopick mrc file

        local step_size=opt.step_size
        if opt.type =='cuda' then deepModel.model:cuda() end

        local autopick_file_total={}
        local process_id = parallel.id
        --parallel.print('Im a worker, going to pick, my ID is: ' .. parallel.id)
        for Index=process_id,pick_number,process_number do
		parallel.print("start to pick mrc file id:"..Index)
                -- read mrc file
                local basename = paths.basename(mrc_file_autopick[Index])
                local dirname = paths.dirname(mrc_file_autopick[Index])
                local pre_basename=string.sub(basename,1,string.len(basename)-4)
                local pixel = readmrc(mrc_file_autopick[Index])

                -- preprocess
                -- scale the input mrc to reduce
                if opt.bin then
                    pixel = scale_model:forward(pixel)
                end

                -- histogram equalization
                if opt.histogram_equalization then
                        pixel = histogram_equalization(pixel)
                end
                -- gaussian lowpass
                if opt.gaussianBlur then
                        pixel = image.convolve(pixel,gaussian_kernel,'same')
                end

                local max = pixel:max()
                local min = pixel:min()
                pixel:add(-min):div(max-min)

                local coordinate_pick,coordinate_cannyEdge,coordinate_postProcess,coordinate_IceClassifier = particle_pick(deepModel,pixel,particle_size,step_size,basename,scale_size,opt)
                table.sort(coordinate_pick,function(a,b) return a[3]>b[3] end)
                if opt.refineCoordinate then
                        local average_refine_length1 = particle_pick_refine(deepModel,pixel,coordinate_pick,particle_size,step_size*2,scale_size,opt)
                end
                -- rescale the coordinate
                for i=1,#coordinate_pick do
                	coordinate_pick[i][1] = math.ceil(coordinate_pick[i][1]*scale)
                	coordinate_pick[i][2] = math.ceil(coordinate_pick[i][2]*scale)
                end
                for i=1,#coordinate_cannyEdge do
                	coordinate_cannyEdge[i][1] = math.ceil(coordinate_cannyEdge[i][1]*scale)
                	coordinate_cannyEdge[i][2] = math.ceil(coordinate_cannyEdge[i][2]*scale)
                end
                for i=1,#coordinate_postProcess do
                	coordinate_postProcess[i][1] = math.ceil(coordinate_postProcess[i][1]*scale)
                	coordinate_postProcess[i][2] = math.ceil(coordinate_postProcess[i][2]*scale)
                end
                for i=1,#coordinate_IceClassifier do
                	coordinate_IceClassifier[i][1] = math.ceil(coordinate_IceClassifier[i][1]*scale)
                	coordinate_IceClassifier[i][2] = math.ceil(coordinate_IceClassifier[i][2]*scale)
                end
		local coordinate = {}
		table.insert(coordinate,coordinate_pick)
		table.insert(coordinate,coordinate_cannyEdge)
		table.insert(coordinate,coordinate_postProcess)
		table.insert(coordinate,coordinate_IceClassifier)
                table.insert(autopick_file_total,coordinate)
        end
        parallel.parent:send(autopick_file_total)
        parallel.print('Finish the job.')
end


-- write the coordinate file 
function autoPicker:write_coordinate(mrc_coordinate,dirname,symbol)
	local symbol = symbol or 'relion'
	local particle_size = self.opt.particle_size
	print("Number of coordinates of files to be written:",#mrc_coordinate)
	for i=1,#mrc_coordinate do
		if #mrc_coordinate[i]>0 then
		--print("filename:",mrc_coordinate[i][1][4])
                local m,n = string.find(mrc_coordinate[i][1][4],"%.mrc")
		if symbol == 'eman' then
                	local coordinate_filename = paths.concat(dirname,string.sub(mrc_coordinate[i][1][4],1,m-1)..'_CNNpick.box')
                	local fp_coor = torch.DiskFile(coordinate_filename,"w")
                	for k=1,#mrc_coordinate[i] do
                        	local line = torch.FloatStorage({mrc_coordinate[i][k][1]-math.ceil(particle_size/2),mrc_coordinate[i][k][2]-math.ceil(particle_size/2),particle_size,particle_size})
                        	fp_coor:writeFloat(line)
                	end
                	fp_coor:close()
		elseif symbol == 'relion' then
                	local coordinate_filename = paths.concat(dirname,string.sub(mrc_coordinate[i][1][4],1,m-1)..'_CNNpick.star')
                	local fp_coor = torch.DiskFile(coordinate_filename,"w")
			local header = "data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
			fp_coor:writeString(header)
			for k=1,#mrc_coordinate[i] do
                        	local line = torch.FloatStorage({mrc_coordinate[i][k][1],mrc_coordinate[i][k][2]})
                        	fp_coor:writeFloat(line)
			end
                	fp_coor:close()
		else
			error("Wrong symbol of writing the coordinate, should be relion|eman, not ",symbol)
		end
		end
        end
end

-- only used for test
function autoPicker:write_test_PR(autopick_total,total_manualpick,total_results_file,graph_name)
	local total_tp = {}
       	local total_recall = {}
       	local total_precision = {}
       	local total_probability = {}
       	local average_distance = {}
       	local total_distance = 0
       	local tem=0 -- current true positive number

       	local index= 0
      	for i=1,#autopick_total do
      		-- compute recall,precision
            	index = index+1
     		if autopick_total[i][5] == 1 then
              		tem = tem+1
              		total_distance = total_distance+autopick_total[i][6]
           	end
            	local precision = tem/index
           	local recall = tem/total_manualpick
       		table.insert(total_tp,tem)
            	table.insert(total_recall,recall)
               	table.insert(total_precision,precision)
             	table.insert(total_probability,autopick_total[i][3])
             	-- compute average distance
            	-- if the particle have a paired manual pick particle
		local average=0
              	if tem > 0 then average = total_distance/tem end
                table.insert(average_distance,average)
    	end
        local fp = torch.DiskFile(total_results_file,"w")
        fp:writeString(table.concat(total_tp,' ')..'\n')
        fp:writeString(table.concat(total_recall,' ')..'\n')
       	fp:writeString(table.concat(total_precision,' ')..'\n')
    	fp:writeString(table.concat(total_probability,' ')..'\n')
    	fp:writeString(table.concat(average_distance,' ')..'\n')
    	fp:writeString('#total autopick number:'..#autopick_total..'\n')
     	fp:writeString('#total manual pick number:'..total_manualpick..'\n')
   	fp:writeString('#the first row is number of true positive\n')
    	fp:writeString('#the second row is recall\n')
       	fp:writeString('#the third row is precision\n')
    	fp:writeString('#the fourth row is probability\n')
      	fp:writeString('#the fiveth row is distance\n')

      	print('---test for sort--')
     	print('total_autopick:',#autopick_total)
     	print('total_manualpick:',total_manualpick)
      	local timesOfmanual = math.ceil(#autopick_total/total_manualpick)
      	for i=1,timesOfmanual do
        	print('autopick_total sort, take the head number of total_manualpick * ratio '..i)
           	fp:writeString('#autopick_total sort, take the head number of total_manualpick * ratio '..i..'\n')
            	if i==timesOfmanual then
              		print('precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall])
              		fp:writeString('#precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall]..'\n')
           	else
                   	print('precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i])
                   	fp:writeString('#precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i]..'\n')
              	end
     	end
      	fp:close()

	-- plot the graph
	local plot_length = 3*total_manualpick
     	if plot_length>#autopick_total then plot_length=#autopick_total end
       	local recall=torch.DoubleTensor(plot_length)
      	local precision=torch.DoubleTensor(plot_length)
      	local probability=torch.DoubleTensor(plot_length)
      	local averageDistance=torch.DoubleTensor(plot_length)
    	local x_value=torch.DoubleTensor(plot_length)
      	for i=1,plot_length do
      		recall[i]=total_recall[i]
       		precision[i]=total_precision[i]
         	probability[i]=total_probability[i]
            	averageDistance[i]=average_distance[i]/50
           	x_value[i]=i/total_manualpick
     	end
       	gnuplot.pngfigure(graph_name)
       	gnuplot.plot({'probability',x_value,probability,'-'},{'recall',x_value,recall,'-'},{'precision',x_value,precision,'-'},{'distance',x_value,averageDistance,'-'})
       	gnuplot.ylabel('distance is normalized by 50')
       	gnuplot.xlabel('manual pick number:'..total_manualpick)
     	gnuplot.grid(true)
      	gnuplot.title('Recall-Precision-Probability-Distance')
     	gnuplot.plotflush()
end

-- do the picking process
function particle_pick(deepModel,pixel,particle_size,step_size,mrc_filename,scale_size,opt)
	local model=deepModel.model
	local m,n = string.find(mrc_filename,"%.%mrc")
	local base = paths.concat(opt.debugDir,string.sub(mrc_filename,1,m-1))
	local draw_filename0 = base.."_classifier.jpg"
	local draw_filename1 = base.."_classifier_filter.jpg"
	local postProcess_pixel=pixel:clone()
	local image_col = pixel:size(2)
	local image_row = pixel:size(3)

	local output_y = math.floor((image_col-particle_size)/step_size)
	local output_x = math.floor((image_row-particle_size)/step_size)

        local output = torch.Tensor(1, output_y, output_x)
        output:zero()

        local y_index=1
        local time1 = sys.clock()
        for particle_y = 1, image_col-particle_size,step_size do
                local x_index=1 
                for particle_x = 1,image_row-particle_size,step_size do
                        local particle = pixel:narrow(2,particle_y,particle_size):narrow(3,particle_x,particle_size):clone()
                        -- preprocess
                        --normalization
                        particle:float()
                        local max = particle:max()
                        local min = particle:min()
                        particle:add(-min):div(max-min)
    
                        --scale to the size fit to the model
                        particle = image.scale(particle,scale_size[2],scale_size[3])
                        -- sum mean and div std
                        local mean = particle:mean()
                        local std = particle:std()
                        particle:add(-mean)
                        particle:div(std)
    
    
                        -- classifier
                        if opt.type == 'double' then particle = particle:double()
                        elseif opt.type == 'cuda' then particle = particle:cuda() 
                        elseif opt.type == 'float' then particle = particle:float() end 
                        local pred = model:forward(particle)
                        output[1][y_index][x_index] = math.exp(pred[1])
                        x_index = x_index+1
                end
                y_index = y_index+1
        end

        --[[
        -- batch input
        local inputs = torch.Tensor(output_y*output_x, scale_size[1], scale_size[2], scale_size[3])

	local y_index = 0
        local index = 0
	local time1 = sys.clock()
        
	for particle_y = 1, image_col-particle_size,step_size do
		for particle_x = 1,image_row-particle_size,step_size do
			local particle = pixel:narrow(2,particle_y,particle_size):narrow(3,particle_x,particle_size):clone()
			-- preprocess
			--normalization
			particle:float()
			local max = particle:max()
			local min = particle:min()
			particle:add(-min):div(max-min)
			
			--scale to the size fit to the model
			particle = image.scale(particle, scale_size[2], scale_size[3])
			-- sum mean and div std
			local mean = particle:mean()
			local std = particle:std()
			particle:add(-mean)
			particle:div(std)
                        index = index + 1
                        inputs[{{index}}] = particle
		end
		y_index = y_index+1
	end

        predictData = {}
        predictData.data = inputs
        predictData.size = inputs:size(1)
        prediction = deepModel:prediction(predictData, 1000, opt)
        print(prediction:size())

	local output = torch.Tensor(1, output_y, output_x):copy(prediction[{{},{1}}])
        --]]

	local time2 = sys.clock()	
	if opt.visualize then
		--print("In function particle_pick,scan the mrc time cost:",(time2-time1)/60)
		local draw_output = output:clone()
		for i=1,draw_output:size(2) do
		for j=1,draw_output:size(3) do
			if draw_output[1][i][j]<opt.binaryThreshold then
				draw_output[1][i][j] = 0
			end
		end
		end 
		image.save(draw_filename0,draw_output) 
	end

	local time3 = sys.clock()
	if opt.deleteIceByConnectedArea then
		output[{1}] = bwlabel(output:clone(),math.ceil(particle_size/step_size),opt)
	        if opt.visualize then 
		        --print("In function particle_pick,bwlabel the mrc time cost:",(time4-time3)/60)
		        local draw_output = output:clone()
		        for i=1,draw_output:size(2) do
		        for j=1,draw_output:size(3) do
			        if draw_output[1][i][j]<opt.binaryThreshold then
				        draw_output[1][i][j] = 0
			        end
		        end
		        end 
		        image.save(draw_filename1,draw_output) 
	        end
	end
	local time4 = sys.clock()
	
	-- pick local peak in a fixed window
	local window_size= math.ceil(particle_size/step_size*opt.minDistanceBetweenParticleRate)
	
	-- every coordinate point  #1=coor_x #2=coor_y #3=probability #4=mrc file name #5=isMatch #6=distance
	local coordinate_pick_small = pickLocalPeak(output:clone(),window_size,mrc_filename)
	local time5 = sys.clock()
	--print("In function particle_pick,pickLocalPeak the mrc time cost:",(time5-time4)/60)

	-- recover the cooridnate, abolish the effect of step_size scale
	local coordinate_pick={}
	for i=1,#coordinate_pick_small do
		local target_x = coordinate_pick_small[i][1]
		local target_y = coordinate_pick_small[i][2]
		local coor_x = math.ceil((target_x-1)*step_size+1+particle_size/2)
		local coor_y = math.ceil((target_y-1)*step_size+1+particle_size/2)
		
		local point = {}
		table.insert(point,coor_x) 	
		table.insert(point,coor_y) 	
		table.insert(point,coordinate_pick_small[i][3]) 	
		table.insert(point,coordinate_pick_small[i][4])
		table.insert(coordinate_pick,point)
	end
	
	-- using the Canny edge detection to get rid of ice noise
	local coordinate_cannyEdge_negative = {}
	if opt.EdgeDetect then
		local coordinate_cannyEdge_positive = {}
		local mask_ice,scale_mask_ice = canny_edge(pixel:clone(),particle_size,opt.debugDir,base,opt.particle_edge_notable,opt.carbonFilmDetect)
		if mask_ice then	
			for i=1,#coordinate_pick do
				local ny = math.ceil(coordinate_pick[i][2]/scale_mask_ice)
				local nx = math.ceil(coordinate_pick[i][1]/scale_mask_ice)
				if mask_ice[ny][nx] == 1 then
					table.insert(coordinate_cannyEdge_negative,coordinate_pick[i])
				else
					table.insert(coordinate_cannyEdge_positive,coordinate_pick[i])
				end
			end
			coordinate_pick = coordinate_cannyEdge_positive
		end
		local time6=sys.clock()
		--print("In function particle_pick, Canny edge detection time cost:",(time6-time5)/60)
	end
		
	-- using a post-process to delete ice
	local coordinate_postProcess_negative = {}
	if opt.postProcess then
		local coordinate_postProcess_positive = {}
		postProcess_pixel=wienerFilter(postProcess_pixel,9)
		local mean=postProcess_pixel:mean()
		local std=postProcess_pixel:std()
		local ice_threshold_lowpixel = mean-3*std
		local number_of_highscore = 0
		local low_pixel = {} 
                for i = 1,#coordinate_pick do
			if coordinate_pick[i][3]>0 then
				number_of_highscore = number_of_highscore+1
				local particle_x= coordinate_pick[i][1]-math.ceil(particle_size/2)
				local particle_y= coordinate_pick[i][2]-math.ceil(particle_size/2)
                        	local particle = postProcess_pixel:narrow(2,particle_y,particle_size):narrow(3,particle_x,particle_size):clone()
				local low_pixel_number=0
				for j=1,particle:size(2) do
				for k=1,particle:size(3) do
					if particle[1][j][k]<ice_threshold_lowpixel then
						low_pixel_number = low_pixel_number+1
					end	
				end
				end
				local pair={}
				pair[1]=i
				pair[2]=low_pixel_number
				table.insert(low_pixel,pair)
                	end
		end
		if number_of_highscore > 0 then		
			local low_pixel_tenor=torch.Tensor(number_of_highscore)
			for i=1,#low_pixel do
				low_pixel_tenor[i]=low_pixel[i][2]
			end
			local mean=low_pixel_tenor:mean()
			local std=low_pixel_tenor:std()
			local threshold_ice = mean+opt.postProcessStd*std
			local number_of_ice = 0
			for i=1,#low_pixel do
				local index=low_pixel[i][1]
				if low_pixel[i][2]>threshold_ice then
					table.insert(coordinate_postProcess_negative,coordinate_pick[index])
					number_of_ice=number_of_ice+1
				else
					table.insert(coordinate_postProcess_positive,coordinate_pick[index])
				end
			end
			coordinate_pick = coordinate_postProcess_positive
		end	
	end
 
	-- using another classifier to delete ice 	
	local coordinate_IceClassifier_negative={}
	if opt.IceClassifier ~= 'none'  then 
		local i=1	
		local coordinate_IceClassifier_positive={}
		for i=1,#coordinate_pick do
			local coor_x = coordinate_pick[i][1]
			local coor_y = coordinate_pick[i][2]
			local model2 = torch.load(opt.IceClassifier)
			if opt.type == 'cuda' then model2:cuda() end
			local particle = pixel:narrow(2,coor_y-particle_size/2,particle_size):narrow(3,coor_x-particle_size/2,particle_size):clone()
		  	local max = particle:max()
			local min = particle:min()
			particle:add(-min):div(max-min)	
			--scale to the size fit to the model
			particle = image.scale(particle,scale_size[2],scale_size[3])
			-- sum mean and div std
			local mean = particle:mean()
			local std = particle:std()
			particle:add(-mean)
			particle:div(std)
			-- classifier
			if opt.type == 'double' then particle = particle:double()
			elseif opt.type == 'cuda' then particle = particle:cuda() 
			elseif opt.type == 'float' then particle = particle:float() end
			local pred = model2:forward(particle)
	
			if(math.exp(pred[1])<0.5) then
				-- save the coordinate classified as ice
				table.insert(coordinate_IceClassifier_negative,coordinate_pick[i])
			else
				table.insert(coordinate_IceClassifier_positive,coordinate_pick[i])
			end
		end
		coordinate_pick = coordinate_IceClassifier_positive
	end
	return coordinate_pick, coordinate_cannyEdge_negative,coordinate_postProcess_negative,coordinate_IceClassifier_negative
end

-- refine the coordinate center 
function particle_pick_refine(deepModel,pixel,coordinate_pick,particle_size,search_length,scale_size,opt)
	local model=deepModel.model
	if math.ceil(search_length%2)==0 then search_length = search_length+1 end
        local image_col = pixel:size(2)
        local image_row = pixel:size(3)
        local time1 = sys.clock()
	local average_length_change = 0
	local particle_number = 0
	for i=1,#coordinate_pick do
		if coordinate_pick[i][3]>=0.5 then
        		local output = torch.Tensor(search_length,search_length)
        		output:zero()
			local coor_x = math.ceil(coordinate_pick[i][1])
			local coor_y = math.ceil(coordinate_pick[i][2])
			for j=1,search_length do
			for k=1,search_length do
        			local particle_y = coor_y-math.floor(search_length/2)+j  --center
                		local particle_x = coor_x-math.floor(search_length/2)+k  --center
				particle_x = particle_x-math.floor(particle_size/2)	
				particle_y = particle_y-math.floor(particle_size/2)
				if particle_x>0 and particle_y>0 and (particle_x+particle_size-1)<=image_row and (particle_y+particle_size-1)<=image_col then
                        		local particle = pixel:narrow(2,particle_y,particle_size):narrow(3,particle_x,particle_size):clone()
                        		-- preprocess
                        		--normalization
                        		particle:float()
                        		local max = particle:max()
                        		local min = particle:min()
                        		particle:add(-min):div(max-min)

                        		--scale to the size fit to the model
                        		particle = image.scale(particle,scale_size[2],scale_size[3])
                        		-- sum mean and div std
                        		local mean = particle:mean()
                        		local std = particle:std()
                        		particle:add(-mean)
                        		particle:div(std)

                        		-- classifier
                        		if opt.type == 'double' then particle = particle:double()
                        		elseif opt.type == 'cuda' then particle = particle:cuda()
                        		elseif opt.type == 'float' then particle = particle:float() end
                        		local pred = model:forward(particle)
                        		output[j][k] = math.exp(pred[1])
				end
			end
			end
			local max = output:max()
			if max<coordinate_pick[i][3] then print("Difference,original score:"..coordinate_pick[i][3].." refine:"..max) end
			local average_x = 0
			local average_y = 0
			local number = 0
			for j=1,search_length do
			for k=1,search_length do
        			local particle_y = coor_y-math.floor(search_length/2)+j  --center
                		local particle_x = coor_x-math.floor(search_length/2)+k  --center
				if output[j][k]== max then
					number = number+1
					average_x = average_x+particle_x
					average_y = average_y+particle_y
				end 
			end
			end
			--if number>1 then print("In function particle_pick_refine, same point:",number) end
			average_x = math.ceil(average_x/number)
			average_y = math.ceil(average_y/number)
			coordinate_pick[i][1]=average_x
			coordinate_pick[i][2]=average_y
			average_length_change = average_length_change+math.sqrt((average_x-coor_x)^2+(average_y-coor_y)^2)
			particle_number = particle_number+1
                end
        end
	if particle_number == 0 then
		average_length_change = 0
	else
		average_length_change = average_length_change/particle_number
	end
        local time2 = sys.clock()
        --print("In function particle_pick_refine(refine score above 0.5),time cost:",(time2-time1)/60)
	return average_length_change			
end

-- display the autopick results in image
function autoPicker:display(autopick_file_total,symbol)
	for i=1,#autopick_file_total do
		if #autopick_file_total[i]>0 then
			local mrc_filename = autopick_file_total[i][1][4]
			if paths.basename(self.mrc_file_autopick[i]) ~= mrc_filename then
				error("Wrong, no match for mrc file and autopick coordinate!")
			end
			local pre_basename = string.sub(mrc_filename,1,string.len(mrc_filename)-4)
			local pixel=readmrc(self.mrc_file_autopick[i])
			pixel = self.scale_model:forward(pixel)
			local max = pixel:max()
			local min = pixel:min()
			pixel:add(-min):div(max-min)
			local filename_autopick_final = paths.concat(self.opt.debugDir,pre_basename.."_"..symbol..".jpg")
			-- scale the coordinate
			for j=1,#autopick_file_total[i] do
				autopick_file_total[i][j][1] = math.ceil(autopick_file_total[i][j][1]/self.scale)
				autopick_file_total[i][j][2] = math.ceil(autopick_file_total[i][j][2]/self.scale)	
			end 
			display(pixel,autopick_file_total[i],self.particle_size,filename_autopick_final,"blue")
		
			-- re-scale the coordinate
			for j=1,#autopick_file_total[i] do
				autopick_file_total[i][j][1] = math.ceil(autopick_file_total[i][j][1]*self.scale)
				autopick_file_total[i][j][2] = math.ceil(autopick_file_total[i][j][2]*self.scale)	
			end
		end 
	end
end
