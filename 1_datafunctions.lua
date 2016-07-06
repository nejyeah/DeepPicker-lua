require 'image'
require 'nn'
require '1_datafunctions_preprocess'

function table.deepcopy(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local new_table = {}
        lookup_table[object] = new_table
        for index, value in pairs(object) do
            new_table[_copy(index)] = _copy(value)
        end
        return setmetatable(new_table, getmetatable(object))
    end
    return _copy(object)
end

--random flip, random rotate the image in eight fixed angle 
function jitter(s)
    local d=torch.rand(3)
    -- vflip
    if d[1] > 0.5 then
        s = image.vflip(s)
    end
    -- hflip
    if d[2] > 0.5 then
        s = image.hflip(s)
    end
    -- rotation
    local angle=math.ceil(d[3]/0.125)
    s=image.rotate(s,math.pi*angle/8)
    return s
end

--random flip, random rotate the image in eight fixed angle 
function jitter_evaluation(im)
    local o = torch.Tensor(32, im:size(1), im:size(2), im:size(3))
    local d=torch.rand(3)
    rotate(im, o[{ {1, 8}, {}, {}, {}  }])
    -- vflip
    s = image.vflip(im)
    rotate(s, o[{ {9, 16}, {}, {}, {}  }])
    -- hflip
    s = image.hflip(im)
    rotate(s, o[{ {17, 24}, {}, {}, {}  }])
    -- vflip and hflip
    s = image.vflip(s)
    rotate(s, o[{ {25, 32}, {}, {}, {}  }])
    return o:clone() 
end

function rotate(im, o)
    o[1] = im
    o[2] = image.rotate(im, math.pi*1/8)
    o[3] = image.rotate(im, math.pi*2/8)
    o[4] = image.rotate(im, math.pi*3/8)
    o[5] = image.rotate(im, math.pi*4/8)
    o[6] = image.rotate(im, math.pi*5/8)
    o[7] = image.rotate(im, math.pi*6/8)
    o[8] = image.rotate(im, math.pi*7/8)
end

function trim(str)
    local index=string.find(str,"%S")
    if index then
        str = string.sub(str,index,#str)		
    end
    return str
end	

function split(str,delim)
    local res = {}
    while true do
        if str and string.find(str,"%S") then 
            str = trim(str)
            local nextpos = string.find(str, delim)
            if not nextpos then
                local item = string.sub(str, 1, #str)
                res[#res+1] = item
                str = nil
            else
                local item = string.sub(str, 1, nextpos-1)
                res[#res+1] = item
                str = string.sub(str,nextpos+1,#str)
            end
        else
            break
        end
    end
    return res
end

function readmrc(filename)
	--define the mrc_header
	local mrc_header={}
	local size1 = 10
	local part1 = torch.IntStorage(size1):fill(0)
	--[[
	--elements in part1
	int nx; //number of columns (fastest changing in map)
    	int ny;  //number of rows 
    	int nz;  //number of sections (slowest changing in map)
    	int mode;  //MODE     data type :
              // 0       image : signed 8-bit bytes range -128 to 127
              // 1       image : 16-bit halfwords
              // 2       image : 32-bit float
              // 3       transform : complex 16-bit integers
              // 4       transform : complex 32-bit reals
              // 6       image : unsigned 16-bit range 0 to 65535
    	int nxstart;  //number of first column in map (Default = 0)
    	int nystart;  //number of first row in map
    	int nzstart;  //number of first section in map
    	int mx;  // number of intervals along X   
    	int my;  //number of intervals along Y    
    	int mz;  //number of intervals along Z
	--]]

	local size2 = 6
	local part2 = torch.FloatStorage(size2):fill(0.0)
	--[[
	--elements in part2
	float cella[3];  //cell dimensions in angstroms   
    	float cellb[3];  //cell angles in degrees 
	--]]
	
	local size3 = 3	
	local part3 = torch.IntStorage(size3):fill(0)
	--[[
	--elements in part3
	int mapc;  //axis corresp to cols (1,2,3 for X,Y,Z)    
    	int mapr;  //axis corresp to rows (1,2,3 for X,Y,Z)    
    	int maps;  // axis corresp to sections (1,2,3 for X,Y,Z)
	--]]

	local size4 = 3
	local part4 = torch.FloatStorage(size4):fill(0.0)
	--[[
	--elements in part4
	float dmin;  //minimum density value    
    	float dmax;  //maximum density value    
    	float dmean;  //mean density value
	--]]
	
	local size5 = 2
	local part5 = torch.IntStorage(size5):fill(0)
	--[[
	--elements in part5
	int ispg;  //space group number 0 or 1 (default=0)    
    	int nsymbt;  //number of bytes used for symmetry data (0 or 80)
	--]]

	local size6 = 100
	local part6 = torch.CharStorage(size6):fill(0)
	--[[
	--elements in part6
	char extra[100];  //extra space used for anything   - 0 by default
	--]]

	local size7 = 3
	local part7 = torch.FloatStorage(size7):fill(0.0)
	--[[
	--elements in part7
	float origin[3];  //origin in X,Y,Z used for transforms
	--]]

	local size8 = 4
	local part8 = torch.CharStorage(size8):fill(0)
	--[[
	--elements in part8
	char map[4];  //character string 'MAP ' to identify file type
	--]]

	local size9 = 1
	local part9 = torch.IntStorage(size9):fill(0)
	--[[
	--elements in part9
	int machst;  //machine stamp 
	--]]

	local size10 = 1
	local part10 = torch.FloatStorage(size10):fill(0.0)
	--[[
	--elements in part10
	float rms;  //rms deviation of map from mean density
	--]]

	local size11 = 1
	local part11 = torch.IntStorage(size11):fill(0)
	--[[
	--elements in part11
	int nlabels;  //number of labels being used 
	--]]
	
	local size12 = 800
	local part12 = torch.CharStorage(size12):fill(0)
	--[[
	--elements in part12
	 char label[10][80];  //ten 80-character text labels
                          //Symmetry records follow - if any - stored as text 
                          //as in International Tables, operators separated 
                          //by * and grouped into 'lines' of 80 characters 
                          //(ie. symmetry operators do not cross the ends of 
                          //the 80-character 'lines' and the 'lines' do not 
                          //terminate in a *). 
                          //Data records follow.
	--]]

	local fp= torch.DiskFile(filename,"r")
	fp:binary()
	--local isbig = fp:isBigEndianCPU()
	--local islittle = fp:isLittleEndianCPU()
	--print("isbigEndianCPU:",isbig)
	--print("isLittleEndianCPU:",islittle)
	--fp:bigEndianEncoding()
	--isbig = fp:isBigEndianCPU()
	--print("isbigEndianCPU:",isbig)
	
	--read mrc header
	local read_size1 = fp:readInt(part1)	
	local read_size2 = fp:readFloat(part2)	
	local read_size3 = fp:readInt(part3)	
	local read_size4 = fp:readFloat(part4)	
	local read_size5 = fp:readInt(part5)	
	local read_size6 = fp:readChar(part6)	
	local read_size7 = fp:readFloat(part7)	
	local read_size8 = fp:readChar(part8)	
	local read_size9 = fp:readInt(part9)	
	local read_size10 = fp:readFloat(part10)	
	local read_size11 = fp:readInt(part11)	
	local read_size12 = fp:readChar(part12)
	if read_size1~=size1 or read_size2~=size2 or read_size3~=size3 or read_size4~=size4	or read_size5~=size5 or read_size6~=size6 or read_size7~=size7 or read_size8~=size8 or read_size9~=size9 or read_size10~=size10 or read_size11~=size11 or read_size12~=size12 then 
		print('read mrcfile'..filename..' header wrong!')
	end
	
	table.insert(mrc_header,part1)		
	table.insert(mrc_header,part2)		
	table.insert(mrc_header,part3)		
	table.insert(mrc_header,part4)		
	table.insert(mrc_header,part5)		
	table.insert(mrc_header,part6)		
	table.insert(mrc_header,part7)		
	table.insert(mrc_header,part8)		
	table.insert(mrc_header,part9)		
	table.insert(mrc_header,part10)		
	table.insert(mrc_header,part11)		
	table.insert(mrc_header,part12)

    	--int nsymbt;  //number of bytes used for symmetry data (0 or 80)
	--read the additional data
	--print("dmin:",part4[1])  --minimum density value    
    	--print("dmax:",part4[2])  --maximum density value    
    	--print("dmean:",part4[3])  --mean density value
	--print('mrc_row:',part1[1])
	--print('mrc_col:',part1[2])
	--print('mode:',part1[4])
	--print("symmetry data:",part5[2])
	local part_symmetry
	local read_size13
	if part5[2] ~= 0 then
		part_symmetry = torch.CharStorage(part5[2])
		read_size13 = fp:readChar(part_symmetry)
	end
	--start to read mrc pixels content
	local mrc_row = part1[1]
	local mrc_col = part1[2]
	local mode = part1[4]
	local mrc_size = mrc_row*mrc_col
	local buffer
	if mode ==0 then 
		buffer = fp:readChar(mrc_size)
	elseif mode == 1 then
		buffer = fp:readShort(mrc_size)
	elseif mode == 2 then		
		buffer = fp:readFloat(mrc_size)
	elseif mode == 3 then
		print('can not process '..filename.. ', for the mode is 3, data type is complex 16-bit integers!')
	elseif mode == 4 then 
		print('can not process '..filename.. ', for the mode is 4, data type is complex 32-bit integers!')
	elseif mode == 6 then 
		buffer = fp:readShort(mrc_size)
	else 
		print('Unknown mode in '..filename..', not the 0|1|2|3|4|6 .')
	end	
	fp:close()
	local pixel = torch.Tensor(1,mrc_col,mrc_row)
	pixel:float()
	local s = pixel:storage()
	s:copy(buffer)    -- save a lot time compared to assign value through for loop
	local max = pixel:max()
	local min = pixel:min()
	pixel:add(-min):div(max-min)
	return pixel:clone()
end

function read_coordinate_relion(coordinatefile)
 	local fp = assert(io.open(coordinatefile,"r"))
	local flag_data = false
	local site_rlnCoordinateX = 0
	local site_rlnCoordinateY = 0
	local site_flag = 0
	local coordinate = {}
	--print('-------------coordinatefile-------------')
	while true do
        	local line = fp:read("*line")
 		if line == nil then break end
 		if string.find(line,"^%_") then
 			local label = string.match(line,"%_(%w+)")
 			flag_data = true
 			site_flag = site_flag+1
			if label == "rlnCoordinateX" then site_rlnCoordinateX = site_flag
 			elseif label == "rlnCoordinateY" then   site_rlnCoordinateY = site_flag
 			elseif label == "rlnMicrographName" then site_rlnMicrographName = site_flag
 			end
 		elseif flag_data and string.find(line,"%w+") then
 			local values
			--print(line)
 			values = split(line,"%s+")
			local coordinateX = values[site_rlnCoordinateX]
                        local coordinateY = values[site_rlnCoordinateY]
			local point={}
			point[1]=tonumber(coordinateX)
			point[2]=tonumber(coordinateY)
			point[3]=0
                        point[4]=0
                        point[5]=0
                        point[6]=0
                        point[7]=0
                        point[8]=0
			table.insert(coordinate,point)
		end	
	end
	fp:close()
	return coordinate	
end

function read_coordinate_eman(coordinatefile)
	local fp= torch.DiskFile(coordinatefile,"r")
	fp:quiet()
	local coordinate = {}
	while true do
		local line = fp:readString("*l")
                local number1,number2,number3,number4 = string.match(line,"(%d+)%s+(%d+)%s+(%d+)%s+(%d+)")
		local number5 = string.match(line,"(%d%.%d+)%s*")
                if number1~=nil and number2~=nil and number3 ~=nil then
                        local point = {}
                        table.insert(point,tonumber(number1+number3/2))
                        table.insert(point,tonumber(number2+number3/2))
			if number5 ~=nil then 
				number5 = tonumber(number5)
				table.insert(point,number5)
			end
			point[3]=0
			point[4]=0
			point[5]=0
			point[6]=0
			point[7]=0
			point[8]=0	
                       	table.insert(coordinate,point)
                end
                if fp:hasError() then break end
	end
	fp:close()
	return coordinate
end

-- extract the particles from the mrc based on coordinate(center)
function pickout_particle(mrc_pixel,coordinate,particle_size)
	local mrc_y = mrc_pixel:size(2)
	local mrc_x = mrc_pixel:size(3)
	local particle_number = #coordinate
	local particle = torch.Tensor(particle_number,1,particle_size,particle_size)
	local index=0
	local half_particle_size = math.ceil(particle_size/2)
	--print("particle_number:",particle_number)	
	for i=1,particle_number do
		local coor_x = coordinate[i][1]
		local coor_y = coordinate[i][2]
		--print("coor_x:"..coor_x.." coor_y:"..coor_y)
		if coor_x>half_particle_size and coor_y>half_particle_size and coor_x+half_particle_size < mrc_x and coor_y+half_particle_size<mrc_y then
			index = index+1
			particle[index] = mrc_pixel:narrow(2,coor_y-half_particle_size,particle_size):narrow(3,coor_x-half_particle_size,particle_size):clone()
			local max = particle[index]:max()
			local min = particle[index]:min()
			particle[index]:add(-min):div(max-min)
		end
	end
	if index == 0 then
		return nil
	else
		particle = particle[{ {1,index},{1} }]	
		return particle:clone()
	end		
end

--random extract the negative particle from the mrc based on the positive particle coordinate
function pickout_negative_particle(mrc_pixel,positive_coordinate,particle_size,threshlod)
	local threshlod = threshlod or 0.6
	--print('threshlod:'..threshlod)
	local mrc_y = mrc_pixel:size(2)
	local mrc_x = mrc_pixel:size(3)
	local particle_number = #positive_coordinate
	local negative_coordinate = {}
	local particle = torch.Tensor(particle_number,1,particle_size,particle_size)
	for i=1,particle_number do
		local coorx=0
		local coory=0
		while true do
			coorx = torch.random()%(mrc_x-particle_size)+1
			coory = torch.random()%(mrc_y-particle_size)+1
			local index = 0 
			for j=1,particle_number do
				index=j
				local distance = math.sqrt( (coorx+math.ceil(particle_size/2)-positive_coordinate[j][1])^2+(coory+math.ceil(particle_size/2)-positive_coordinate[j][2])^2 )
				if distance<threshlod*particle_size then break end
			end
			if index==particle_number then break end
		end
		local tem = {}
		table.insert(tem,coorx+math.ceil(particle_size/2))
		table.insert(tem,coory+math.ceil(particle_size/2))
		table.insert(negative_coordinate,tem)
		particle[i] = mrc_pixel:narrow(2,coory,particle_size):narrow(3,coorx,particle_size):clone()
		local max = particle[i]:max()
		local min = particle[i]:min()
		particle[i]:add(-min):div(max-min)
	end
	return particle:clone(),negative_coordinate	
end

  
function sort_correlation(neighbour)
	local sortFunc = function(a,b) return a[2]<b[2] end
	table.sort(neighbour,sortFunc)
end
function test_correlation(coordinate_pick,coordinate_manual,threshold)
	if coordinate_pick ==nil or coordinate_manual==nil then 
		error("Wrong coordinate!")
	end
	local tp_sigle = 0
	local average_distance = 0 
	
	for j=1,#coordinate_manual do
		coordinate_manual[j][5] = 0
                local coor_x = coordinate_manual[j][1]
                local coor_y = coordinate_manual[j][2]
		local neighbour = {} 
                for k=1,#coordinate_pick do
		      	if coordinate_pick[k][5]==0 then
                      		local coor_mx = coordinate_pick[k][1]
                      		local coor_my = coordinate_pick[k][2]
                      		local abs_x = math.abs(coor_mx-coor_x)
                      		local abs_y = math.abs(coor_my-coor_y)
                      		local length = math.sqrt(math.pow(abs_x,2)+math.pow(abs_y,2)) 
                      		if length < threshold then
			   		local same_n = {}
			   		table.insert(same_n,k)
			   		table.insert(same_n,length)
			   		table.insert(neighbour,same_n)
                      		end
		   	end
               	end
		if #neighbour>=1 then
			if #neighbour>1 then
				sort_correlation(neighbour)
			end	
			average_distance = average_distance+neighbour[1][2]
			local index = neighbour[1][1]
			coordinate_pick[index][5] = 1
			coordinate_pick[index][6] = neighbour[1][2]
			coordinate_pick[index][7] = coor_x
			coordinate_pick[index][8] = coor_y
			tp_sigle = tp_sigle+1 
			coordinate_manual[j][5] = 1
		end
        end
	average_distance = average_distance/tp_sigle
	return tp_sigle,average_distance
end

-------------------------------------------------------------------------------------------------------------------
-- binary the output score map and abolish the large connected domain
function bwlabel(pixel,scale_particle_size,opt)
	if pixel:dim()==3 then
		pixel = pixel[1]:clone()
	end
	local data = pixel:clone()
	local col=data:size(1)
	local row=data:size(2)
	local s=data:storage()
	for i=1,s:size() do
		if s[i]<opt.binaryThreshold then
			s[i] = 0
		else
			s[i] = 1
		end
	end
	local stRun,enRun,rowRun,NumberOfRuns = fillRunVectors(data)
	if #stRun == 0 then
		return pixel:clone()
	end

	local equivalences,runLabels = firstPass(stRun,enRun,rowRun,NumberOfRuns)
	local runLabels,number = replaceSameLabel(runLabels,equivalences)
	local areaOfconnected = computeArea(pixel,stRun,enRun,rowRun,runLabels)
	local number = 0
	for i=1,areaOfconnected:size(1) do
		if areaOfconnected[i]>10 then
			number = number+1
		end
	end
	if number > 20 then
	local areaOfconnected_large = torch.Tensor(number):fill(0)
	local j=1
	
	for i=1,areaOfconnected:size(1) do
		if areaOfconnected[i]>10 then
			areaOfconnected_large[j] = areaOfconnected[i]
			j = j+1
		end
	end
	local mean = areaOfconnected_large:mean()
	local std = areaOfconnected_large:std()
	local high_threshold = mean+std*opt.meanRate

	local numer_large_area = 0
	for i=1,areaOfconnected:size(1) do
		if areaOfconnected[i]> high_threshold then 
			numer_large_area = numer_large_area + 1
		end
	end
	print("Delete large connected area: ",numer_large_area)
	pixel = zeroOfAreaOutsideThreshold(pixel,areaOfconnected,stRun,enRun,rowRun,runLabels,high_threshold)
	end
	return pixel:clone()
end

------------------------------------------------------------------------------------------------------------------------------
-- local peak detection to choose the center of particles 
-- window_size define the size of local area
-- threshold define the probability threshold used to choose particle 
function pickLocalPeak(output,window_size,mrc_filename)
	if output:dim()==3 then
		output = output[1]:clone()
	end
	local clone_output = output:clone()
	local output_y=clone_output:size(1)
	local output_x=clone_output:size(2)

	local time1 = sys.clock()
	for coor_y=1, output_y do
        for coor_x=1, output_x do
             local target_pixel = output[coor_y][coor_x]
             for j=1,window_size do
             for i=1,window_size do
                 clone_y = coor_y-math.ceil(window_size/2)-1+j
                 clone_x = coor_x-math.ceil(window_size/2)-1+i
                 if clone_y>0 and clone_x>0 and clone_y<=output_y and clone_x<=output_x and clone_output[clone_y][clone_x]<target_pixel then
                      clone_output[clone_y][clone_x] = target_pixel
                 end
             end
             end
        end
        end
	local time2 = sys.clock()

	-- pick out the particles coordinate
        local coordinate_pick_small_all = {}
        local coordinate_x = 0
        local coordinate_y = 0
        for coor_y=1, output_y do
        for coor_x=1 , output_x do
            -- choose one from  the same largest values
            local target_pixel = output[coor_y][coor_x]
            if target_pixel~=0 and target_pixel == clone_output[coor_y][coor_x] then
            	local number = 0
		local average_y=0
		local average_x=0
                for j=1,window_size do
                for i=1,window_size do
                     local clone_y = coor_y-math.ceil(window_size/2)-1+j
                     local clone_x = coor_x-math.ceil(window_size/2)-1+i
                     if clone_y>0 and clone_x>0 and clone_y<=output_y and clone_x<=output_x and output[clone_y][clone_x]==target_pixel then
                           number = number+1
			   average_y = average_y+clone_y
			   average_x = average_x+clone_x
                           clone_output[clone_y][clone_x]= 0
                     end
                end
                end
		average_y=math.ceil(average_y/number)
		average_x=math.ceil(average_x/number)
		local point = {}
                table.insert(point,average_x)           -- 1,
                table.insert(point,average_y)		-- 2,
		table.insert(point,target_pixel)	-- 3, value
		table.insert(point,mrc_filename)	-- 4,
		table.insert(point,number)  		-- 5, number of peak
		table.insert(point,0)  			-- 6, symbol
		table.insert(coordinate_pick_small_all,point)
             end
        end
        end
	local time3 = sys.clock()
	
        -- abolish close points
        for i=1,#coordinate_pick_small_all do
                if coordinate_pick_small_all[i][6]==0 then
                	for j=i+1,#coordinate_pick_small_all do
				if coordinate_pick_small_all[i][6] == 1 then break end
                        	if coordinate_pick_small_all[j][6] == 0 then
                                	local d_x=coordinate_pick_small_all[i][1]-coordinate_pick_small_all[j][1]
                                	local d_y=coordinate_pick_small_all[i][2]-coordinate_pick_small_all[j][2]
                                	local d_distance=math.sqrt(d_x^2+d_y^2)
                                	if d_distance<window_size/2 then
						if coordinate_pick_small_all[i][3]>=coordinate_pick_small_all[j][3] then
                                        		coordinate_pick_small_all[j][6]=1
						else coordinate_pick_small_all[i][6]=1
						end
                                	end
                        	end
                	end
                end
        end
        local coordinate_pick_small = {}
        for i=1,#coordinate_pick_small_all do
                local point={}
                if coordinate_pick_small_all[i][6]==0 then
                        table.insert(point,coordinate_pick_small_all[i][1])  --1, coor_x
                        table.insert(point,coordinate_pick_small_all[i][2])  --2, coor_y
                        table.insert(point,coordinate_pick_small_all[i][3])  --3, probability
                        table.insert(point,coordinate_pick_small_all[i][4])  --4, mrc_filename
                        table.insert(coordinate_pick_small,point)
                end
        end
	local time4 = sys.clock()
	return coordinate_pick_small
end

--------------------------------------------------------------------------------------------------------------------------------
function read_star(star_file, label_table)
    local dirname = paths.dirname(star_file)
    local inp = assert(io.open(star_file,"r"))
    local flag_data = false
    local label_all = {}
    local value_all = {}
    while true do
        local line = inp:read("*line")
        if line == nil then break end
        if string.find(line,"^%_") then
            local label = string.match(line,"%_(%w+)")
            flag_data = true
            table.insert(label_all, label)
        elseif flag_data and string.find(line,"%w+") then
            local values = {}
            line = trim(line)
            values = split(line,"%s+")
            table.insert(value_all, values)
        end
    end
    
    local label_index = {}
    for i=1, #label_table do
        for j=1, #label_all do
            if label_table[i] == label_all[j] then
                 label_index[i] = j
                 break
            end
        end
        if label_index[i] == nil then error("Invalide label:"..label_table[i]) end
    end
    local value_all_sample = {}
    for i=1, #value_all do
        local value_each_sample = {}
        for j=1, #label_index do
            table.insert(value_each_sample, value_all[i][ label_index[j] ])
        end
        table.insert(value_all_sample, value_each_sample)
    end
    return value_all_sample
end
-- some interface of reading STAR file of Relion
-- return three table, one contains mrcfile names, and the other contain coordinates(a two stage table)
-- mrc_files,coordinate,total_particle_number = read_star_coordinate(star_file_table)
-- 
function read_star_coordinate(star_file_table)
	local mrc_files= {}
	local coordinate = {}
	local total_particle = 0
	for i=1,#star_file_table do
        	local starfile = star_file_table[i]
        	print('read file '..i..'/'..#star_file_table..' :'..starfile)
		local dirname = paths.dirname(starfile)
        	local inp = assert(io.open(starfile,"r"))
        	local flag_data = false
        	local number = 1
        	local site_flag = 0
        	local site_rlnCoordinateX = 0
        	local site_rlnCoordinateY = 0
        	local site_rlnMicrographName = 0

        	while true do
                	local line = inp:read("*line")
                	if line == nil then break end
                	if string.find(line,"^%_") then
                        	local label = string.match(line,"%_(%w+)")
                        	flag_data = true
                        	site_flag = site_flag+1
                        	if label == "rlnCoordinateX" then site_rlnCoordinateX = site_flag
                        	elseif label == "rlnCoordinateY" then   site_rlnCoordinateY = site_flag
                        	elseif label == "rlnMicrographName" then site_rlnMicrographName = site_flag
                        	end
                	elseif flag_data and string.find(line,"%w+") then
                        	total_particle = total_particle+1
                        	local values = {}
				line = trim(line)
				values = split(line,"%s+")
                        	local coordinateX = values[site_rlnCoordinateX]
                        	local coordinateY = values[site_rlnCoordinateY]
                        	local micrographName = paths.concat(dirname,values[site_rlnMicrographName])
                        	local exist = false
                        	local exist_index = 0
                        	for j=1,#mrc_files do
                                	if mrc_files[j] == micrographName then
                                        	exist=true
                                        	exist_index = j
                                       	 	break
                                	end
                        	end
                        	if exist then
					local coor = {}
					table.insert(coor,coordinateX)
					table.insert(coor,coordinateY)
                                	table.insert(coordinate[exist_index],coor)
                        	else
                                	mrc_files[#mrc_files+1] = micrographName
                                	local coor_M = {}
					local coor_c = {}
                                	table.insert(coor_c,coordinateX)
                                	table.insert(coor_c,coordinateY)
					
                                	table.insert(coor_M,coor_c)
                                	coordinate[#coordinate+1] = coor_M
                        	end
                	end
                	number= number+1
        	end
		--print("rlnCoordinateX:",site_rlnCoordinateX)
		--print("rlnCoordinateY:",site_rlnCoordinateY)
		--print("rlnMicrographName:",site_rlnMicrographName)
	end
	return mrc_files,coordinate,total_particle
end

function read_star_particles(star_file_table, particle_size, bin_scale, needNegative, mrcfile_dir, debugDIR)
	if not needNegative then needNegative = false end	
	local mrc_files, coordinate, total_particle_number = read_star_coordinate(star_file_table)
	
	local mrc_file_all = {}
	local coordinate_all = {}
	local particle_number_all = 0
	local dirname = paths.dirname(mrc_files[1])
	if mrcfile_dir then dirname=mrcfile_dir end
	if paths.dirp(dirname) then
		for i=1,#mrc_files do
			local mrc_file
			if mrcfile_dir then
				local basename=paths.basename(mrc_files[i])
                		mrc_file = paths.concat(dirname,basename)
			else
                		mrc_file = mrc_files[i]
			end
        		if paths.filep(mrc_file) then 
                                table.insert(mrc_file_all,mrc_file)
				table.insert(coordinate_all,coordinate[i])
				particle_number_all = particle_number_all+#coordinate[i]
			else
				error("Can't find file:",mrc_file)
			end
                end
	else
		error("Wrong in function read_star_particles(star_file_table,mrcfile_dir,particle_size,scale_size,pick_negative),not a dir:",dirname)
	end
	
	local data = readmrc(mrc_file_all[1])
	-- preprocess
	--scale the input mrc to reduce
	local col = data:size(2)
	local row = data:size(3)
        -- need to be manuallly set, the default is 3
        local scale = bin_scale
        --[[
	local scale = 1
	if col>row then
       		scale = math.floor(row/1000)
	else
       		scale = math.floor(col/1000)
	end
        --]]
		
	local scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
	scale_model.weight:fill(1)
	scale_model.bias:fill(0)
	--print(particle_size)
	particle_size = math.ceil(particle_size/scale)
	local particle_table_positive = {}
	local particle_table_negative = {}
	for i=1,#mrc_file_all do
		print('read mrc file index '..i..'/'..#mrc_file_all..' ')
		local basename = paths.basename(mrc_file_all[i])
		local base = string.sub(basename,1,string.len(basename)-4)
                data = readmrc(mrc_file_all[i])
                data = scale_model:forward(data)
                local max = data:max()
                local min = data:min()
                data:add(-min):div(max-min)

                local coordinate_data = coordinate_all[i]
                for j=1,#coordinate_data do
                        coordinate_data[j][1] = math.ceil(coordinate_data[j][1]/scale)
                        coordinate_data[j][2] = math.ceil(coordinate_data[j][2]/scale)
                end
                local positive_particle = pickout_particle(data,coordinate_data,particle_size)
                if positive_particle then
			table.insert(particle_table_positive,positive_particle)
			if needNegative then
				local negative_particle,neg_coordinate = pickout_negative_particle(data,coordinate_data,particle_size)	 
				local filename_particle=paths.concat(debugDIR,base.."_train_particle.jpg")
                                --print(filename_particle)
                                display_compare(data,coordinate_data,neg_coordinate,particle_size,filename_particle)

				table.insert(particle_table_negative,negative_particle)
			end
                end 
        	collectgarbage()
	end

	local positive_data = torch.Tensor(particle_number_all,1,particle_size,particle_size)
	local start_p = 1
	local end_p = 0
	for i=1,#mrc_file_all do
        	-- positive 
        	end_p = end_p+particle_table_positive[i]:size(1)
        	positive_data[{{start_p,end_p} }] = particle_table_positive[i]:clone()
        	start_p = start_p+particle_table_positive[i]:size(1)
	end
	if needNegative then
		local negative_data = torch.Tensor(particle_number_all,1,particle_size,particle_size)
		local start_n = 1
		local end_n = 0
		for i=1,#mrc_file_all do
        		-- negative
        		end_n = end_n+particle_table_negative[i]:size(1)
        		negative_data[{{start_n,end_n} }] = particle_table_negative[i]:clone()
        		start_n = start_n+particle_table_negative[i]:size(1)
		end
		return positive_data:clone(),negative_data:clone()
	else
		return positive_data:clone()
	end
end 

function get_test_coordinate_pick(coordinate_pick,test_threshold)
	local test_coordinate_pick = {}
	for i=1,#coordinate_pick do
		if coordinate_pick[i][3]> test_threshold then
			local point = {}
			table.insert(point,coordinate_pick[i][1])
			table.insert(point,coordinate_pick[i][2])
			table.insert(test_coordinate_pick,point)
		end
	end
	return test_coordinate_pick
end

function get_head(coordinate_pick,flag_threshold)
	local test_coordinate_pick = {}
	if flag_threshold>1 then
		if flag_threshold > #coordinate_pick then flag_threshold = #coordinate_pick end
		for i=1,flag_threshold do
			table.insert(test_coordinate_pick,coordinate_pick[i])
		end
	elseif flag_threshold>=0 then
		for i=1,#coordinate_pick do
			if coordinate_pick[i][3]<flag_threshold then break end
			if coordinate_pick[i][3]>=flag_threshold then
				table.insert(test_coordinate_pick,coordinate_pick[i])
			end
		end
	else
		error('wrong parameters in function get_head(coordinate_pick,flag_threshold),flag_threshold must be positive')
	end	
	return test_coordinate_pick
end

-- called when write the coordinte to file finally
function process_coordinate(coordinate_pick)
	local mrc_coordinate = {}
	for i=1,#coordinate_pick do
		local mrc_filename = coordinate_pick[i][4]
		local exist = false
		for j=1,#mrc_coordinate do
			if mrc_filename==mrc_coordinate[j][1][4] then
				table.insert(mrc_coordinate[j],coordinate_pick[i])
				exist = true
				break
			end
		end
		if not exist then
			local coordinate = {}
			table.insert(coordinate,coordinate_pick[i])
			table.insert(mrc_coordinate,coordinate)
		end
	end
	return mrc_coordinate
end
------------------------------------------------------------------------------------------

function showPNP(testPNP,filenamePN,filenameNP)
        local number_testPN = table.getn(testPNP[1])
        local number_testNP = table.getn(testPNP[2])
        local testPN = torch.Tensor(number_testPN,1,scale_size[2],scale_size[3])
        local testNP = torch.Tensor(number_testNP,1,scale_size[2],scale_size[3])

        if number_testPN ~=0 then
                for i=1,number_testPN do
                        testPNP[1][i]:float()
                        for n=1,scale_size[2] do
                        for m=1,scale_size[3] do
                                testPN[i][1][n][m] = testPNP[1][i][1][n][m]
                        end
                        end
                end
                testPN:float()
                local number_row = math.ceil(math.sqrt(number_testPN))
                local testPN_image = image.toDisplayTensor{input=filenamePN,padding = 3,nrow = number_row}
                image.save(filename,testPN_image)
        end

        if number_testNP ~=0 then 
                for i=1,number_testNP do
                        testPNP[2][i]:float()
                        for n=1,scale_size[2] do
                        for m=1,scale_size[3] do
                                testNP[i][1][n][m] = testPNP[2][i][1][n][m]
                        end
                        end
                end
                testNP:float()
                local number_row = math.ceil(math.sqrt(number_testNP))
                local testNP_image = image.toDisplayTensor{input=filenameNP,padding = 3,nrow = 8}
                image.save(filename,testNP_image)
        end
end

---------------------------------------------------------------------------------------------------
-- preprocess, delete ice
-- preprocess, wiener filtering
function wienerFilter(data,size)
        local data_copy=data:clone()
        local map_std=data:std()
        local row=data:size(2)
        local col=data:size(3)
        local half_length=math.floor(size/2)
        for i=1,row-size do
        for j=1,col-size do
                local patch=data_copy:narrow(2,i,size):narrow(3,j,size)
                local std=patch:std()
                local mean=patch:mean()
                data[1][i+half_length][j+half_length]=mean+(std^2-map_std^2)/(std^2)*(data_copy[1][i+half_length][j+half_length]-mean)
        end
        end
        return data
end

-----------------------------------------------------------------------------------------------------
-- write the coordinate file 
function write_coordinate(mrc_coordinate, dirname, particle_size, coordinate_type, coordinate_symbol)
        local coordinate_type = coordinate_type or 'relion'
        local coordinate_symbol = coordinate_symbol or '_CNNpick'
        for i=1,#mrc_coordinate do
                local m,n = string.find(mrc_coordinate[i][1][4],"%.mrc")
                if coordinate_type == 'eman' then
                        local coordinate_filename = paths.concat(dirname,string.sub(mrc_coordinate[i][1][4],1,m-1)..coordinate_symbol..'.box')
                        local fp_coor = torch.DiskFile(coordinate_filename,"w")
                        for k=1,#mrc_coordinate[i] do
                                local line = torch.FloatStorage({mrc_coordinate[i][k][1]-math.ceil(particle_size/2),mrc_coordinate[i][k][2]-math.ceil(particle_size/2),particle_size,particle_size})
                                fp_coor:writeFloat(line)
                        end
                        fp_coor:close()
                elseif coordinate_type == 'relion' then
                        local coordinate_filename = paths.concat(dirname,string.sub(mrc_coordinate[i][1][4],1,m-1)..coordinate_symbol..'.star')
                        local fp_coor = torch.DiskFile(coordinate_filename,"w")
                        local header = "data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
                        fp_coor:writeString(header)
                        for k=1,#mrc_coordinate[i] do
                                local line = torch.FloatStorage({mrc_coordinate[i][k][1],mrc_coordinate[i][k][2]})
                                fp_coor:writeFloat(line)
                        end
                        fp_coor:close()
                else
                        error("Wrong coordinate type of writing the coordinate, should be relion|eman :",coordinate_type)
                end
        end
end
