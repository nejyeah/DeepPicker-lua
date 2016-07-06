require 'image'
require 'torch'
require 'nn'
require 'paths'

--------------------------------------------------------
-- connected component analysis

-- expand the binary matrix with range n
-- set the neighbor(range n) of active point(value=1) to 1
-- input:
--	data: a bianry matrix[M][N]
--	n: lenth of neighbor to be set to 1
function expand_connectedArea(data,n)
        col = data:size(1)
        row = data:size(2)
        expand = torch.IntTensor(col+2*n,row+2*n):fill(0)
        data:int()
        expand[{ {n+1,col+n},{n+1,row+n} }] = data:clone()
        for i=1,col do
        for j=1,row do
                if data[i][j] == 1 then
                        expand[{ {i,2*n+i},{j,2*n+j} }]:add(1)
                end
        end
        end
        data = expand[{ {n+1,col+n},{n+1,row+n} }]:clone()
        for i=1,col do
        for j=1,row do
                if data[i][j]>=1 then data[i][j]=1
                else data[i][j]=0 end
        end
        end
        return data
end

-- compute the connected pixel in each row
-- input:
--	pixel: a binary matrix[M][N]
function fillRunVectors(pixel)
        local stRun = {}
        local enRun = {}
        local rowRun = {}
        local NumberOfRuns = 0
        local col=pixel:size(1)
        local row=pixel:size(2)
        for i=1,col do
                if(pixel[i][1]==1) then
                        NumberOfRuns = NumberOfRuns+1
                        table.insert(stRun,1)
                        table.insert(rowRun,i)
                end
                for j=2,row do
                        if pixel[i][j-1]==0 and pixel[i][j] ==1 then
                                NumberOfRuns = NumberOfRuns+1
                                table.insert(stRun,j)
                                table.insert(rowRun,i)
                        elseif pixel[i][j-1]==1 and pixel[i][j] ==0 then
                                table.insert(enRun,j-1)
                        end
                end
                if pixel[i][row]==1 then
                        table.insert(enRun,row)
                end
        end
        return stRun,enRun,rowRun,NumberOfRuns
end

-- assign labels to each run and the label equivalence relationships generated 
-- input:
--	stRun,enRun,rowRun,NumberOfRuns: the output of the function fillRunVectors()
function firstPass(stRun,enRun,rowRun,NumberOfRuns)
        local runLabels = torch.IntTensor(NumberOfRuns):fill(0)
        local equivalences = {}
        local idxLabel = 1
        local curRowIdx = 1
        local firstRunOnCur = 1
        local firstRunOnPre = 1
        local lastRunOnPre = 0
	local consistent_Inrow = false
        for i=1,NumberOfRuns do
                if rowRun[i] ~=curRowIdx then
			if rowRun[i] == curRowIdx+1 then
				consistent_Inrow = true
			else
				consistent_Inrow = false
			end 
                        curRowIdx = rowRun[i]
                        firstRunOnPre = firstRunOnCur
                        lastRunOnPre = i-1
                        firstRunOnCur = i
                end
                -- if the row not consistent, then there is no need to compare with last row
		if consistent_Inrow then
                  	for j=firstRunOnPre,lastRunOnPre do
                                if stRun[i] <= enRun[j]+1 and enRun[i]>=stRun[j]-1 then
                                        if runLabels[i] ==0 then
                                                runLabels[i] = runLabels[j]
                                        elseif runLabels[i] ~= runLabels[j] then
                                                table.insert(equivalences,runLabels[i])
                                                table.insert(equivalences,runLabels[j])
                                        end
                                end
                        end
		end
                if runLabels[i] == 0 then
                        runLabels[i] = idxLabel
                        idxLabel = idxLabel+1
                end
        end
        return equivalences,runLabels
end

-- replace the same label and re-arrange the labels
-- first creat a label equivalences matrix 
-- iterative searching for the largest label equivalences list
-- input: 
--	output of the function firstPass()
function search(sameMinFlag,index,flag,labelMatrix)
        local data = labelMatrix[index]
        flag[index] = sameMinFlag
        for j=1,data:size(1) do
                if data[j] ==1 and flag[j]==0 then
                        search(sameMinFlag,j,flag,labelMatrix)
                end
        end
end
function replaceSameLabel(runLabels,equivalences)
        -- iteration method
        local number = runLabels:max()
        local labelMatrix = torch.IntTensor(number,number):fill(0)
        for i=1,#equivalences/2 do
                labelMatrix[ equivalences[2*i-1] ][ equivalences[2*i] ] = 1
                labelMatrix[ equivalences[2*i] ][ equivalences[2*i-1] ] = 1
        end
        local flag = torch.IntTensor(number):fill(0)
        local flagIndex = 1
        for i=1,number do
                if flag[i] ==0  then
                        search(flagIndex,i,flag,labelMatrix)
                        flagIndex = flagIndex+1
                end
        end
        for i=1,runLabels:size(1) do
                runLabels[i] = flag[runLabels[i] ]
        end
        return runLabels,flagIndex-1
end

-- estimate the area of each connected component
-- input:
--	pixel: the binary matrix or the original probability matrix(only estimate the area, the probility of which larger than 0.5)
--	stRun,enRun,rowRun,runLabels: output of the previous functions
function computeArea(pixel,stRun,enRun,rowRun,runLabels)
        local numberOfConnectedArea = runLabels:max()
        local boundary = {}
        local area = torch.Tensor(numberOfConnectedArea):fill(0)
        for i=1,runLabels:size(1) do
                local symbol = runLabels[i]
                if not boundary[symbol] then
                        boundary[symbol] = {}
                        boundary[symbol][1] = pixel:size(2)  --x min
                        boundary[symbol][2] = pixel:size(1)  --y min
                        boundary[symbol][3] = 0  --x max
                        boundary[symbol][4] = 0  --y max
                 end
                 for j=stRun[i],enRun[i] do
                        if pixel[ rowRun[i] ][j]>0.5 then
                                area[symbol] = area[symbol]+1
                                -- update the boundary information
                                if rowRun[i] < boundary[symbol][2] then boundary[symbol][2] = rowRun[i] end
                                if rowRun[i] > boundary[symbol][4] then boundary[symbol][4] = rowRun[i] end
                                if j < boundary[symbol][1] then boundary[symbol][1] = j end
                                if j > boundary[symbol][3] then boundary[symbol][3] = j end
                        end
                end

        end
        return area,boundary
end

-- zero the connected components which the area is either larger than a upper_threshold or smaller than a lower_threshold
-- input:
--	pixel: the binary matrix or the original probability matrix
--	areaOfconnected,stRun,enRun,rowRun,runLabels: output of the previous functions
--	upper_threshold: nil or a positive int value 
--	lower_threshold: nil or a positive int value
function zeroOfAreaOutsideThreshold(pixel,areaOfconnected,stRun,enRun,rowRun,runLabels,upper_threshold,lower_threshold)
	if upper_threshold then
        	for i=1,areaOfconnected:size(1) do
                	if areaOfconnected[i] > upper_threshold then
                        	for j=1,runLabels:size(1) do
                                	if runLabels[j] == i then
                                        	pixel:narrow(1,rowRun[j],1):narrow(2,stRun[j],enRun[j]-stRun[j]+1):fill(0)
                                        	rowRun[j] = 0
                                        	stRun[j] = 0
                                        	enRun[j] = 0
                                        	runLabels[j] = 0
                                	end
                        	end
                	end
        	end
	end
	if lower_threshold then
        	for i=1,areaOfconnected:size(1) do
                	if areaOfconnected[i] < lower_threshold then
                        	for j=1,runLabels:size(1) do
                                	if runLabels[j] == i then
                                        	pixel:narrow(1,rowRun[j],1):narrow(2,stRun[j],enRun[j]-stRun[j]+1):fill(0)
                                        	rowRun[j] = 0
                                        	stRun[j] = 0
                                        	enRun[j] = 0
                                        	runLabels[j] = 0
                                	end
                        	end
                	end
        	end
	end
        return pixel:clone()
end

-- zero the connected components which the length is either larger than a upper_threshold or smaller than a lower_threshold
-- input:
--	pixel: the binary matrix or the original probability matrix
--	areaOfconnected,stRun,enRun,rowRun,runLabels: output of the previous functions
--	upper_threshold: nil or a positive int value 
--	lower_threshold: nil or a positive int value
function zeroOfLengthOutsideThreshold(pixel,boundary,stRun,enRun,rowRun,runLabels,upper_threshold,lower_threshold)
	if upper_threshold then
        	for i=1,#boundary do
			local length = math.sqrt((boundary[i][1]-boundary[i][3])^2+(boundary[i][2]-boundary[i][4])^2)
                	if length > upper_threshold then
                        	for j=1,runLabels:size(1) do
                                	if runLabels[j] == i then
                                        	pixel:narrow(1,rowRun[j],1):narrow(2,stRun[j],enRun[j]-stRun[j]+1):fill(0)
                                        	rowRun[j] = 0
                                        	stRun[j] = 0
                                        	enRun[j] = 0
                                        	runLabels[j] = 0
                                	end
                        	end
                	end
        	end
	end
	if lower_threshold then
        	for i=1,#boundary do
			local length = math.sqrt((boundary[i][1]-boundary[i][3])^2+(boundary[i][2]-boundary[i][4])^2)
                	if length < lower_threshold then
                        	for j=1,runLabels:size(1) do
                                	if runLabels[j] == i then
                                        	pixel:narrow(1,rowRun[j],1):narrow(2,stRun[j],enRun[j]-stRun[j]+1):fill(0)
                                        	rowRun[j] = 0
                                        	stRun[j] = 0
                                        	enRun[j] = 0
                                        	runLabels[j] = 0
                                	end
                        	end
                	end
        	end
	end
        return pixel:clone()
end

----------------------------------------------------------
-- least square curve fit
-- f(x) = a*x^2+b*x+c
-- input:
--	coordinate: type table
-- 	coordinate[i]: type table
-- 	coordinate[i][1]: x
--	coordinate[i][2]: y
-- return:
--	a,b,c: the parameters of the curve
function least_squares(coordinate)
        local x_4 = 0
        local x_3 = 0
        local x_2 = 0
        local x_1 = 0
        local x_0 = 0
        local y_x0 = 0
        local y_x1 = 0
        local y_x2 = 0
        for i=1,#coordinate do
                x_4 = x_4 + coordinate[i][1]^4
                x_3 = x_3 + coordinate[i][1]^3
                x_2 = x_2 + coordinate[i][1]^2
                x_1 = x_1 + coordinate[i][1]^1
                x_0 = x_0 + 1

                y_x2 = y_x2 + coordinate[i][2]*coordinate[i][1]^2
                y_x1 = y_x1 + coordinate[i][2]*coordinate[i][1]
                y_x0 = y_x0 + coordinate[i][2]
        end

        local matrix = torch.Tensor(3,3)
        local value_matrix = torch.Tensor(3)
        -- initialize
        matrix[1][1] = x_4
        matrix[1][2] = x_3
        matrix[1][3] = x_2
        matrix[2][1] = x_3
        matrix[2][2] = x_2
        matrix[2][3] = x_1
        matrix[3][1] = x_2
        matrix[3][2] = x_1
        matrix[3][3] = x_0

	value_matrix[1] = y_x2
        value_matrix[2] = y_x1
        value_matrix[3] = y_x0
        -- inverse
        -- normalize the first row
        local i=1 
        local div = matrix[i][1]
        for j=1,3 do
                matrix[i][j] = matrix[i][j]/div
        end
        value_matrix[i] = value_matrix[i]/div
        -- zero the second row
        i = 2 
        local scale = matrix[i][1]
        for j=1,3 do
                matrix[i][j] = matrix[i][j] - scale*matrix[1][j]
        end
        value_matrix[i] = value_matrix[i] - scale*value_matrix[1]
        -- normalize the second row
        div = matrix[i][2]
        for j=1,3 do
                matrix[i][j] = matrix[i][j]/div
        end
        value_matrix[i] = value_matrix[i]/div
        -- zero the third row    
        i = 3
        local scale = matrix[i][1]
        for j=1,3 do
                matrix[i][j] = matrix[i][j] - scale*matrix[1][j]
        end
        value_matrix[i] = value_matrix[i] - scale*value_matrix[1]

        local scale = matrix[i][2]
        for j=1,3 do
                matrix[i][j] = matrix[i][j] - scale*matrix[2][j]
        end
        value_matrix[i] = value_matrix[i] - scale*value_matrix[2]


        div = matrix[i][3]
        for j=1,3 do
                matrix[i][j] = matrix[i][j]/div
        end
        value_matrix[i] = value_matrix[i]/div
        --
        local c = value_matrix[3]
        local b = (value_matrix[2]-c*matrix[2][3])/matrix[2][2]
        local a = (value_matrix[1]-b*matrix[1][2]-c*matrix[1][3])/matrix[1][1]
        return a,b,c
end

-- least square curve fit
-- It will test wether the curve fit f(x) = a*x^2+b*x+c or f(y) = a*y^2+b*y+c
-- input:
--	coordinate: type table
-- 	coordinate[i]: type table
-- 	coordinate[i][1]: x
--	coordinate[i][2]: y
-- return:
--	results : table format
--	results[1] : string 'x' or 'y'
--	results[2] : a
-- 	results[3] : b
--	results[4] : c
function least_squares_refine(coordinate)
	-- y=a*x^2+b*x+c fitting
	-- or 
	-- x = a*y^2+b*y+c fitting
        local number_same_x = 0
        -- estimate the number of point which have the same x axis
        table.sort(coordinate,function(a,b) return a[1]<b[1] end)
        local cur_x = 0
        for i=1,#coordinate do
                if coordinate[i][1] == cur_x then
                        number_same_x = number_same_x + 1
                end
                if coordinate[i][1] ~= cur_x then
                        cur_x = coordinate[i][1] 
                end
        end

        local number_same_y = 0
        -- estimate the number of point which have the same x axis
        table.sort(coordinate,function(a,b) return a[2]<b[2] end)
        local cur_y = 0
        for i=1,#coordinate do
                if coordinate[i][2] == cur_y then
                        number_same_y = number_same_y + 1
                end
                if coordinate[i][2] ~= cur_y then
                        cur_y = coordinate[i][2] 
                end
        end
        --print("number_same_y:",number_same_y)
        --print("number_same_x:",number_same_x)
        if number_same_y>=number_same_x then
                --print('Fit the least square based on the x as variable')
	        local a,b,c = least_squares(coordinate)
	        local results = {}	
		results[1] = 'x'		
		results[2] = a		
		results[3] = b		
		results[4] = c
                return results		
        else
                --print('Fit the least square based on the y as variable')
	        local coordinate_t = {}
	        for i=1,#coordinate do
		        local coor = {}
		        coor[1] = coordinate[i][2]
		        coor[2] = coordinate[i][1]
		        table.insert(coordinate_t,coor)
	        end
	        local a,b,c = least_squares(coordinate_t)
	        local results = {}	
		results[1] = 'y'		
		results[2] = a		
		results[3] = b		
		results[4] = c	
                return results	
        end
end

-----------------------------------------------------------------------------------------
-- Canny edge detection algorithm
-- First, input a porbability matrix and bin the matrix to 400 ~ 800
-- Second, using a gaussian kernel to lowpass the matrix to enforce the edge information
-- third, binary the matrix in different threholds
-- forth, using the connected component analysis to delete the noise edge
-- output a bianry matrix show the edge 
-- input:
--	data: probability matrix[1][M][N], a three dimensional tensor
--	dir: debug information output dir
--	base: symbol of the image name to be output
function canny_edge(data,particle_size,dir,base,particle_edge_notable,carbonFilmExist)
	local col = data:size(2)
        local row = data:size(3)
	local filename1 = paths.concat(dir,base..'_o.jpg')
	--image.save(filename1,data)
	-- scale the image
        local scale = 1
        if col>row then
                scale = math.floor(row/400)
        else
                scale = math.floor(col/400)
        end
	local scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
        scale_model.weight:fill(1)
        scale_model.bias:fill(0)
	data = scale_model:forward(data)
	local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)
	local particle_size = math.ceil(particle_size/scale)
		
	local filename2 = paths.concat(dir,base..'_os.jpg')
	-- used to detect the carbon area 
	local pixel = data:clone()
	--image.save(filename2,data)
	
	-- gaussian lowpass the image
	local time1 = sys.clock()
	local kernel_size = 9
	local gaussianSigma = 0.1
        local gaussian_kernel = image.gaussian(kernel_size,gaussianSigma,1,true)
	data = image.convolve(data,gaussian_kernel,'same')
	local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)		
	local filename3 = paths.concat(dir,base..'_osg.jpg')
	--image.save(filename3,data)
	local time2 = sys.clock()
	--print('time cost for gaussian lowpass:',(time2-time1)/60)

	-- estimate the gradient
	local dx = data:clone():fill(0)
	local dy = data:clone():fill(0)
	local delta = data:clone():fill(0)
	for i=kernel_size,data:size(2)-kernel_size do
	for j=kernel_size,data:size(3)-kernel_size do
		dx[1][i][j] = data[1][i][j+1]-data[1][i][j]
		dy[1][i][j] = data[1][i+1][j]-data[1][i][j]
		delta[1][i][j] = math.sqrt(dx[1][i][j]^2 + dy[1][i][j]^2)
	end
	end
	local filename4 = paths.concat(dir,base..'_osgr.jpg')
	--image.save(filename4,delta)
	local time3 = sys.clock()
	--print('time cost for estimate gradient:',(time3-time2)/60)
	
	-- abolish the non-local maximum
	local maximum_delta = delta:clone():fill(0)
	for i=2,maximum_delta:size(2)-1 do
	for j=2,maximum_delta:size(3)-1 do
	if delta[1][i][j] > 0 then
		local gradTemp1 = 0
		local gradTemp2 = 0
		local grad1 = 0
		local grad2 = 0
		local grad3 = 0
		local grad4 = 0
		if math.abs(dy[1][i][j])>math.abs(dx[1][i][j]) then  -- write
			local weight_x = math.abs(dx[1][i][j])/math.abs(dy[1][i][j]) 
			local weight_y = 1-weight_x 
			grad2 = delta[1][i-1][j]
			grad4 = delta[1][i+1][j]
			if dx[1][i][j]*dy[1][i][j]>=0 then
				grad1 = delta[1][i-1][j-1]
				grad3 = delta[1][i+1][j+1]
			else
				grad1 = delta[1][i-1][j+1]
				grad3 = delta[1][i+1][j-1] 
			end
			gradTemp1 = weight_x*grad1+weight_y*grad2
			gradTemp2 = weight_x*grad3+weight_y*grad4
		else
			local weight_y = math.abs(dy[1][i][j])/math.abs(dx[1][i][j]) 
			local weight_x = 1-weight_y 
			grad1 = delta[1][i][j-1]	
			grad3 = delta[1][i][j+1]	
			if dx[1][i][j]*dy[1][i][j]>=0 then
				grad2 = delta[1][i-1][j-1]
				grad4 = delta[1][i+1][j+1]
			else
				grad2 = delta[1][i+1][j-1]
				grad4 = delta[1][i-1][j+1] 
			end
			gradTemp1 = weight_x*grad1+weight_y*grad2
			gradTemp2 = weight_x*grad3+weight_y*grad4
		end
		if delta[1][i][j]>=gradTemp1 and delta[1][i][j]>=gradTemp2 then
			maximum_delta[1][i][j] = delta[1][i][j]
		end
	end
	end
	end
	local filename5 = paths.concat(dir,base..'_osgrm.jpg')
	--image.save(filename5,maximum_delta)
	local time4 = sys.clock()
	--print('time cost for abolish the non-local maximum:',(time4-time3)/60)
	
	-- binary the gradient
	local mean = maximum_delta:mean()
	local std = maximum_delta:std()
	local threshold_s3 = mean + std*3
	local threshold_s2 = mean + std*2
	local threshold_s1 = mean + std*1
	--print("threshold_s1:",threshold_s1)
	--print("threshold_s2:",threshold_s2)
	--print("threshold_s3:",threshold_s3)
	local col = maximum_delta:size(2)
	local row = maximum_delta:size(3)
	local binary = torch.Tensor(6,col,row):fill(0)
	for i=1,maximum_delta:size(2)-1 do
	for j=1,maximum_delta:size(3)-1 do
		if maximum_delta[1][i][j] > threshold_s3 then 
			binary[3][i][j] = 1 
		end
		if maximum_delta[1][i][j] > threshold_s2 then 
			binary[2][i][j] = 1 
		end
		if maximum_delta[1][i][j] > threshold_s1 then 
			binary[1][i][j] = 1 
		end
	end
	end
	-- threshold = mean+3*std
	local filename5 = paths.concat(dir,base..'_osgrmb3.jpg')
	--image.save(filename5,binary[3])
	-- threshold1 = mean+2*std
	local filename5 = paths.concat(dir,base..'_osgrmb2.jpg')
	--image.save(filename5,binary[2])
	-- threshold2 = mean+1*std
	local filename5 = paths.concat(dir,base..'_osgrmb1.jpg')
	--image.save(filename5,binary[1])
	local time5 = sys.clock()
	--print('binary the gradinet:',(time5-time4)/60)

	-- bwlabel the binary map
	-- delete those small connected domain
	-- threshold = mean+3*std
	local stRun3,enRun3,rowRun3,NumberOfRuns3 = fillRunVectors(binary[3])
        local equivalences3,runLabels3 = firstPass(stRun3,enRun3,rowRun3,NumberOfRuns3)
        local areaOfconnected3 = computeArea(binary[3],stRun3,enRun3,rowRun3,runLabels3)
	binary[{3}] = zeroOfAreaOutsideThreshold(binary[3],areaOfconnected3,stRun3,enRun3,rowRun3,runLabels3,nil,10)
	local filename6 = paths.concat(dir,base..'_osgrmb3r.jpg')
	--image.save(filename6,binary[3])
	-- initialize label
	for i=1,runLabels3:size(1) do
		if runLabels3[i] ~= 0 then
			for j=stRun3[i],enRun3[i] do
				binary[6][ rowRun3[i] ][j] = runLabels3[i]
			end
		end
	end
	stRun3,enRun3,rowRun3,NumberOfRuns3 = fillRunVectors(binary[3])
        equivalences3,runLabels3 = firstPass(stRun3,enRun3,rowRun3,NumberOfRuns3)
	local time6 = sys.clock()
	--print('connected component analysis for st3:',(time6-time5)/60)

	if #stRun3==0 then
		--print("No connected domain in binary st3(rm 10)! ")
		return nil,nil
	end

	-- delete those small connected domain
	-- threshold1 = mean+2*std
	local stRun2,enRun2,rowRun2,NumberOfRuns2 = fillRunVectors(binary[2])
        local equivalences2,runLabels2 = firstPass(stRun2,enRun2,rowRun2,NumberOfRuns2)
        local runLabels2,number2 = replaceSameLabel(runLabels2,equivalences2)
        local areaOfconnected2,boundary2 = computeArea(binary[2],stRun2,enRun2,rowRun2,runLabels2)
	binary[{2}] = zeroOfAreaOutsideThreshold(binary[2],areaOfconnected2,stRun2,enRun2,rowRun2,runLabels2,nil,10)
	local carbonFilm_binary_mask
	if carbonFilmExist then
                carbonFilm_binary_mask = carbonFilm_edgedetect(data[1],binary[2],dir,base)
                if carbonFilm_binary_mask then
	                local filename6 = paths.concat(dir,base..'_carbon_film_mask.jpg')
	                image.save(filename6,carbonFilm_binary_mask)
                end 
        end
	if particle_edge_notable then
		binary[{2}] = zeroOfLengthOutsideThreshold(binary[2],boundary2,stRun2,enRun2,rowRun2,runLabels2,nil,particle_size)
	end
	local filename6 = paths.concat(dir,base..'_osgrmb2r.jpg')
	--image.save(filename6,binary[2])
	-- initialize label
	-- expand std2
	local binary_std2_expand = binary[2]:clone()
	binary_std2_expand = expand_connectedArea(binary_std2_expand,5)
	local stRun2_expand,enRun2_expand,rowRun2_expand,NumberOfRuns2_expand = fillRunVectors(binary_std2_expand)
	if #stRun2_expand ==0 then
		--print("No connected component in binary st2")
		return nil,nil
	end
        local equivalences2_expand,runLabels2_expand = firstPass(stRun2_expand,enRun2_expand,rowRun2_expand,NumberOfRuns2_expand)
        runLabels2_expand = replaceSameLabel(runLabels2_expand,equivalences2_expand)
        local areaOfconnected2_expand,boundary2_expand = computeArea(binary[2],stRun2_expand,enRun2_expand,rowRun2_expand,runLabels2_expand)
	binary[{2}] = zeroOfAreaOutsideThreshold(binary[2],areaOfconnected2_expand,stRun2_expand,enRun2_expand,rowRun2_expand,runLabels2_expand,nil,20)
	
	for i=1,runLabels2_expand:size(1) do
		if runLabels2_expand[i] ~=0 then
			for j=stRun2_expand[i],enRun2_expand[i] do
				binary[5][ rowRun2_expand[i] ][j] = runLabels2_expand[i]
			end
		end
	end
	local time7 = sys.clock()
	--print('connected component analysis for st2:',(time7-time6)/60)

	-- merge the s3 and s2
	local s2_label_left = {}
	for i=1,runLabels3:size(1) do
		if runLabels3[i] ~=0 then
			local y = rowRun3[i]
			local x = stRun3[i]
			local label = binary[5][y][x]
			if label ~=0 then
				local exist = false
				for j=1,#s2_label_left do
					if s2_label_left[j] == label then
						exist = true
						break
					end
				end
				if not exist then table.insert(s2_label_left,label) end
			end
		end
	end
	--print("Number of connected domain left after s2 and s3 merge:",#s2_label_left)
	-- refreash the s2 tensor
	local binary_s2=binary[2]:clone()
	binary[2]:fill(0)
	for i=1,runLabels2_expand:size(1) do
		local label = runLabels2_expand[i]
		local symbol_left = false
		for j=1,#s2_label_left do
			if s2_label_left[j] == label then
				symbol_left = true
				break
			end
		end
		if symbol_left then
			for j=stRun2_expand[i],enRun2_expand[i] do
				binary[2][ rowRun2_expand[i] ][j] = binary_s2[rowRun2_expand[i] ][j]
			end
		end
		
	end
	local filename6 = paths.concat(dir,base..'_osgrmbs3s2.jpg')
	image.save(filename6,binary[2])
	local time8 = sys.clock()
	--print('connected component analysis merge st3 and st2:',(time8-time7)/60)
	--[[
	local carbonFilm_binary_mask
	if carbonFilmExist then
                carbonFilm_binary_mask = carbonFilm_edgedetect(data[1],binary[2],dir,base)
                if carbonFilm_binary_mask then
	                local filename6 = paths.concat(dir,base..'_carbon_film_mask.jpg')
	                image.save(filename6,carbonFilm_binary_mask)
                end 
        end
        --]]
	-- make a mask
	local mask_ice = binary[2]:clone()
	mask_ice = expand_connectedArea(mask_ice,5)
	local filename6 = paths.concat(dir,base..'_osgrambs3s2_mask.jpg')
	image.save(filename6,mask_ice)
        if carbonFilm_binary_mask then
                for i=1,mask_ice:size(1) do
                for j=1,mask_ice:size(2) do
                        if carbonFilm_binary_mask[i][j] == 1 then mask_ice[i][j] =1 end 
                end
                end
        end
	return mask_ice,scale		
end

function carbonFilm_edgedetect(data,binary_matrix,dir,base)	
	-- find the carbon film edge(very hard to judge) 
	local binary_expand = binary_matrix:clone()
	binary_expand = expand_connectedArea(binary_expand,5)
	local stRun,enRun,rowRun,NumberOfRuns = fillRunVectors(binary_expand)
        local equivalences,runLabels = firstPass(stRun,enRun,rowRun,NumberOfRuns)
        local runLabels,number = replaceSameLabel(runLabels,equivalences)
        local areaOfconnected,boundary = computeArea(binary_matrix,stRun,enRun,rowRun,runLabels)

	local max_area = areaOfconnected:max()
	local max_index = 0
	for i=1,areaOfconnected:size(1) do
		if areaOfconnected[i] == max_area then
			max_index = i
		end
	end
	local x_min = boundary[max_index][1]
	local y_min = boundary[max_index][2]
	local x_max = boundary[max_index][3]
	local y_max = boundary[max_index][4]
	-- show the max connected component
	local binary_maxone = binary_matrix:clone():fill(0)
        local point_all_coor={}
	for i=1,runLabels:size(1) do
		if runLabels[i] == max_index then
                        local point={}
			for j=stRun[i],enRun[i] do
				binary_maxone[rowRun[i] ][j] = binary_matrix[rowRun[i] ][j]
                                table.insert(point,j/binary_maxone:size(2))
                                table.insert(point,rowRun[i]/binary_maxone:size(1))
			end
                        table.insert(point_all_coor,point)
		end
	end
	for i=x_min,x_max do
		binary_maxone[y_min][i] = 1
		binary_maxone[y_max][i] = 1
	end
	for j=y_min,y_max do
		binary_maxone[j][x_min] = 1
		binary_maxone[j][x_max] = 1
	end
	local filename6 = paths.concat(dir,base..'_carbonfilm_edge_rec.jpg')
	image.save(filename6,binary_maxone)
	 
	local max_length = math.sqrt((y_max-y_min)^2+(x_max-x_min)^2)
	local ratio_area_length = max_area/max_length
	--print("max area:",max_area)
	--print("max_length:",max_length)
	--print("max_area/max_length:",ratio_area_length)
	if max_area<200 or max_length<200 then
		--print("No carbon for less area or length!!!")
		return nil
	end
        if max_length<300 and ratio_area_length>5 then 
                --print("No carbon for large ratio_area_length!!!")
                return nil
        end
       
        -- least square fit
        local results=least_squares_refine(point_all_coor)
        local binary_curve = binary_matrix:clone():fill(0)
        local col_y = binary_curve:size(1)
        local row_x = binary_curve:size(2)
        -- carbon film mask
        if results then
                local a = results[2]
                local b = results[3]
                local c = results[4]
                -- estimate the point shared by the curve and the
                -- four sides of square
                local numberOftouch = 0
                -- x = 0
                local y = c
                if y>0 and y<1 then numberOftouch = numberOftouch + 1 end
                -- x = 1
                y = a + b + c
                if y>0 and y<1 then numberOftouch = numberOftouch + 1 end
                -- y = 0 
                local tem = b^2-4*a*c
                if tem>0 then
                        local x1 = (-b+math.sqrt(tem))/(2*a)
                        local x2 = (-b-math.sqrt(tem))/(2*a)
                        if x1>=0 and x1<=1 and x2>=0 and x2<=1 then
                                --print('No carbon, wrong fit of the curve, two points in y=0!')
                                return nil
                        end
                        if x2>0 and x2<1 then numberOftouch = numberOftouch+1 end
                        if x1>0 and x1<1 then numberOftouch = numberOftouch+1 end
                end 
                -- y = 1 
                local tem = b^2-4*a*(c-1)
                if tem>0 then
                        local x1 = (-b+math.sqrt(tem))/(2*a)
                        local x2 = (-b-math.sqrt(tem))/(2*a)
                        if x1>=0 and x1<=1 and x2>=0 and x2<=1 then
                                --print('No carbon, wrong fit of the curve, two points in y=1!')
                                return nil
                        end
                        if x2>0 and x2<1 then numberOftouch = numberOftouch+1 end
                        if x1>0 and x1<1 then numberOftouch = numberOftouch+1 end
                end
                if numberOftouch ~=2 then
                        --print("No carbon, wrong fit of curve, numberOftouch not equal to 2:",numberOftouch)
                        return nil
                end 
                if results[1] == 'x' then
                        -- plot the curve
                        for i=1,row_x do
                                local nx = i/row_x
                                local ny = a*nx^2 + b*nx + c
                                if ny>0 and ny<=1 then
                                        local y = math.ceil(ny*col_y)
                                        binary_curve[y][i] = 1
                                end
                        end
                        local filename = paths.concat(dir,base..'_binary_curve.jpg')
                        image.save(filename,binary_curve)
                        
                        -- judge which side is carbon
                        local average_pixel_up = 0
                        local number_point_up = 0
                        local average_pixel_down = 0
                        local number_point_down = 0
                        for i=1,row_x,10 do
                                local nx = i/row_x
                                local ny = a*nx^2 + b*nx + c
                                if ny>0 and ny<=1 then
                                        local y = math.ceil(ny*col_y)
                                        for j=1,col_y do
                                                if j<y then 
                                                        average_pixel_down = average_pixel_down + data[j][i] 
                                                        number_point_down = number_point_down + 1
                                                end
                                                if j>y then 
                                                        average_pixel_up = average_pixel_up + data[j][i] 
                                                        number_point_up = number_point_up + 1
                                                end
                                        end
                                end
                        end
                        average_pixel_up = average_pixel_up/number_point_up
                        average_pixel_down = average_pixel_down/number_point_down
                        if average_pixel_up<average_pixel_down then
                                for i=1,row_x do
                                        local nx = i/row_x
                                        local ny = a*nx^2 + b*nx + c
                                        if ny>0 and ny<=1 then
                                                local y = math.ceil(ny*col_y)
                                                for j=y,col_y do
                                                        binary_curve[j][i] = 1
                                                end
                                        end
                                end
                        else
                                for i=1,row_x do
                                        local nx = i/row_x
                                        local ny = a*nx^2 + b*nx + c
                                        if ny>0 and ny<=1 then
                                                local y = math.ceil(ny*col_y)
                                                for j=1,y do
                                                        binary_curve[j][i] = 1
                                                end
                                        end
                                end
                        end
                        return binary_curve 
                elseif results[1] == 'y' then
                        -- plot the curve
                        for i=1,col_y do
                                local ny = i/col_y
                                local nx = a*ny^2 + b*ny + c
                                if nx>0 and nx<=1 then
                                        local x = math.ceil(nx*row_x)
                                        binary_curve[i][x] = 1
                                end
                        end
                        local filename = paths.concat(dir,base..'_binary_curve.jpg')
                        image.save(filename,binary_curve) 
                        -- judge which side is carbon
                        local average_pixel_up = 0
                        local number_point_up = 0
                        local average_pixel_down = 0
                        local number_point_down = 0
                        for i=1,col_y do
                                local ny = i/col_y
                                local nx = a*ny^2 + b*ny + c
                                if nx>0 and nx<=1 then
                                        local x = math.ceil(nx*row_x)
                                        for j=1,row_x do
                                                if j<x then
                                                        average_pixel_down = average_pixel_down + data[i][j] 
                                                        number_point_down = number_point_down + 1
                                                end
                                                if j>x then
                                                        average_pixel_up = average_pixel_up + data[i][j] 
                                                        number_point_up = number_point_up + 1
                                                end
                                        end       
                                end
                        end
                        average_pixel_up = average_pixel_up/number_point_up
                        average_pixel_down = average_pixel_down/number_point_down
                        if average_pixel_up<average_pixel_down then
                                for i=1,col_y do
                                        local ny = i/col_y
                                        local nx = a*ny^2 + b*ny + c
                                        if nx>0 and nx<=1 then
                                                local x = math.ceil(nx*row_x)
                                                for j=x,row_x do
                                                        binary_curve[i][j] = 1
                                                end
                                        end
                                end
                        else
                                for i=1,col_y do
                                        local ny = i/col_y
                                        local nx = a*ny^2 + b*ny + c
                                        if nx>0 and nx<=1 then
                                                local x = math.ceil(nx*row_x)
                                                for j=1,x do
                                                        binary_curve[i][j] = 1
                                                end
                                        end
                                end

                        end
                        return binary_curve 

                else
                        --print("Strage output of function least_squares_refine") 
                        return nil
                end
        end  
end
