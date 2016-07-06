----------------------------------------------------------
-- Author WangFeng  2016/04/14
-- This script can be used to visualize mrc files in '.jpg' format.
-- Mrc files can be with two different coordinate files, e.g., manualpick and autopick
-- The two different coordinate files can be in Relion(star)|Eman(format) format

-- Input:
--	-input: the dir of mrc files(coordinate files should be in the same dir)
--	-scale: wether scale the jpg to a small size
--	-particle_size: particle size
--	-coorType1: relion|eman
--	-coorSymbol1: if star file is 'stack_001_corr_manual.star', then the coorSymbol should be '_manual'
--		      if star file is 'stack_001_corr.star', then the coorSymbol should be ''
--	-number: number of mrc files to visualize
-- Output:
--	the images of plot particles will be stored in the same directory of the mrc file

--------------------------------------------------------
require 'torch'
require 'image'
require 'nn'

require '1_datafunctions'
require '1_datafunctions_qt'

if not opt then 
	print '==> mrc2jpg options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('mrc file or dir to jpg')
	cmd:text()
	cmd:text('Options:')
	cmd:option('-input','none','the dir of mrc files or mrcfile')
	cmd:option('-scale',false,'scale the image')
	cmd:option('-particle_size',180,'particle size(pixel)')
	cmd:option('-coorType1','relion','the type of coordinate file')
	cmd:option('-coorSymbol1','no','none,means nothing,the symbol of coordinate file')
	cmd:option('-coorType2','relion','the type of coordinate file')
	cmd:option('-coorSymbol2','no','the symbol of coordinate file')
	cmd:option('-number',0,'number of images to show,if equal to 0 then show all images')
	cmd:option('-sameParticleDistanceRate',0.2,'define the same particle when compare autopick results with manualpick')
	opt=cmd:parse(arg or {})
end

local input = opt.input
local index = 1

local mrc_files = {}
if paths.dirp(input) then
	for file in paths.files(input) do
		if string.match(file,"%w+.mrc") then
			local mrc_filename = paths.concat(input,file)
			table.insert(mrc_files,mrc_filename)
		end	
	end
elseif paths.filep(input) and string.match(input,"%w+.mrc") then
	table.insert(mrc_files,input)
else
	error("Wrong input, there is not mrc files in the dir! Or it not a mrc file!")
end
table.sort(mrc_files)

local data = readmrc(mrc_files[1])
-- preprocess
--scale the input mrc to reduce
local col = data:size(2)
local row = data:size(3)
print("col:",col)
print("row:",row)

local particle_size = opt.particle_size
local scale = 1
if col>row then
       scale = math.floor(row/1000)
else
       scale = math.floor(col/1000)
end
local scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
scale_model.weight:fill(1)
scale_model.bias:fill(0)

local particle_size = math.ceil(particle_size/scale)
local number = #mrc_files
if opt.number < 0 then error("wrong value of opt.number,must >= 0 ")
elseif opt.number<number and opt.number~=0 then	number = opt.number end
for i=1,number do
	local mrc_filename = mrc_files[i]
	local outputfilename = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'.jpg'
	print("Index:",i)
	print("Input:",mrc_filename)
	print("Output:",outputfilename)				
	data = readmrc(mrc_filename)
	if opt.scale then
		data = scale_model:forward(data)
	end
	local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)
	local outimage = image.toDisplayTensor{input = data}
	image.save(outputfilename,outimage)
	collectgarbage()
	
	local isExist_coor1 = false 
	local coordinate_filename1
	if opt.coorSymbol1 ~= 'no' then
		if opt.coorType1 == 'relion' then
			if opt.coorSymbol1 == 'none' then
				coordinate_filename1 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'.star'
			else
				coordinate_filename1 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol1..'.star'
			end
			if paths.filep(coordinate_filename1) then
				coordinate1 = read_coordinate_relion(coordinate_filename1)
				isExist_coor1 = true
			end
		elseif opt.coorType1 == 'eman' then
			if opt.coorSymbol1 == 'none' then
				coordinate_filename1 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'.box'
			else
				coordinate_filename1 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol1..'.box'
			end
			if paths.filep(coordinate_filename1) then
				coordinate1 = read_coordinate_eman(coordinate_filename1)
				isExist_coor1 = true
			end
		else
			error("wrong type of coorSymbol,please set it relion|eman")	
		end
		if isExist_coor1 then
			for i=1,#coordinate1 do
				coordinate1[i][1] = math.ceil(coordinate1[i][1]/scale)
				coordinate1[i][2] = math.ceil(coordinate1[i][2]/scale)
			end
			output_filename1 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol1..'.jpg'
			display(data,coordinate1,particle_size,output_filename1,'red')	
		end
	end
	local coordinate_filename2 
	if opt.coorSymbol2 ~= 'no' then 
		local isExist = false 
		if opt.coorType2 == 'relion' then
			if opt.coorSymbol2 == 'none' then
				coordinate_filename2 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'.star'
			else
				coordinate_filename2 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol2..'.star'
			end
			if paths.filep(coordinate_filename1) then
				coordinate2 = read_coordinate_relion(coordinate_filename2)
				isExist = true
			end
		elseif opt.coorType2 == 'eman' then
			if opt.coorSymbol2 == 'none' then
				coordinate_filename2 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'.box'
			else
				coordinate_filename2 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol2..'.box'
			end
			if paths.filep(coordinate_filename1) then
				coordinate2 = read_coordinate_eman(coordinate_filename2)
				isExist = true
			end
		else
			error("wrong type of coorSymbol,please set it relion|eman")	
		end
		if isExist then	
			for i=1,#coordinate2 do
				coordinate2[i][1] = math.ceil(coordinate2[i][1]/scale)
				coordinate2[i][2] = math.ceil(coordinate2[i][2]/scale)
			end	
			local output_filename2 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..opt.coorSymbol2..'.jpg'
			display(data,coordinate2,particle_size,output_filename2,'blue')	
			if opt.coorSymbol1 ~= 'no' and isExist_coor1 then
				local output_filename3 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'_compare.jpg'
				display_compare(data,coordinate1,coordinate2,particle_size,output_filename3,'red','blue')
				
				local output_filename4 = string.sub(mrc_filename,1,string.len(mrc_filename)-4)..'_compare_clean.jpg'
				test_correlation(coordinate1,coordinate2,particle_size*opt.sameParticleDistanceRate)
				display_compare(data,coordinate1,coordinate2,particle_size,output_filename4,'red','blue',true)
			end

		end
	end 
end
