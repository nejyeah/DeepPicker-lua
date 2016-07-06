require '1_datafunctions'
require '1_datafunctions_qt'

InputData={}

-- used for autopick
-- load the mrc file names from the inputDir 
function InputData:loadMrcFiles(inputDir)
    self.mrc_file_all = {}
    if paths.dirp(inputDir) then
        for file in paths.files(inputDir) do
            if string.match(file,"%w+%.mrc") then
                local mrc_file = paths.concat(inputDir,file)
                table.insert(self.mrc_file_all,mrc_file)
            end
        end
    end
    table.sort(self.mrc_file_all)
end

-- Load samples from the relion 2D classificaton results (a star file)
-- inputDIR: the original mrc file director
-- inputFiles: the star file, like PATH/particles_classification2D.star
-- trainNumber: a specific number like(10000)
-- particle_size: particle size of target molecular
function load_Particle_From_Relion_Star(inputFiles, trainNumber, particle_size, bin_scale)
    local positive_data = {}
    local negative_data = {}
    if not string.match(inputFiles,'%.star$') then error("wrong format of input file, must be `.star` for symbol 1:",inputFiles) end
    if paths.filep(inputFiles) then
        local file_table = {}
        table.insert(file_table,inputFiles)
        local positive_data_tensor, negative_data_tensor = read_star_particles(file_table, particle_size, bin_scale, true, nil, debugDir)
        local positive_number = positive_data_tensor:size(1)
        local negative_number = negative_data_tensor:size(1)
        if positive_number>trainNumber then positive_number=trainNumber end
        if negative_number>trainNumber then negative_number=trainNumber end
        for j=1,positive_number do
            table.insert(positive_data,positive_data_tensor[j])
        end
        for j=1,negative_number do
            table.insert(negative_data,negative_data_tensor[j])
        end
    else
        error("Can not find file:",inputFiles)
    end
    return positive_data, negative_data
end

-- Load samples from torch format files (these file come from run 'extractData.lua'), a way to train the muti-molecular CNN model
-- inputDIR: the Dir of the inputfiles
-- inputFiles: t7 format particle file, like 'spliceosome.t7;gammas.t7;trpv1.t7'
-- trainNumber: a specific number like(10000)
-- particle_size: particle size of target molecular
function load_Particle_From_Torch_t7(inputDIR, inputFiles, trainNumber, particle_size)
    local positive_data = {}
    local negative_data = {}
    local protein_data_file = {}
    local files=split(inputFiles,";")
    for i=1,#files do
        local file=files[i]
        if string.match(file,'%.t7$') then
            local filename = paths.concat(inputDIR,file)
            table.insert(protein_data_file,filename)
        else
            error("Wrong format of input file, the tail must be '.t7' for symbol 2:",file)
        end
    end
			
    for i=1,#protein_data_file do
        if paths.filep(protein_data_file[i]) then	
            local protein_data = torch.load(protein_data_file[i])
            local average_number = math.ceil(trainNumber/#protein_data_file)
            if average_number>#protein_data[1] then average_number = #protein_data[1] end
            for j=1, average_number do
                table.insert(positive_data, protein_data[1][j])
            end
	    if average_number>#protein_data[2] then average_number = #protein_data[2] end
            for j=1,average_number do
                table.insert(negative_data, protein_data[2][j])
            end
        else
            error("Can not find file:"..protein_data_file[i])
        end
    end
    return positive_data, negative_data 
end

-- Load samples from the last autopicking results (torch formate file), a way to do iteration training  
-- inputDIR: the original mrc file directory
-- inputFiles: PATH/coordinate_autopick.t7
-- trainNumber: can be a probability threshold(like 0.5) or a specific number like(10000) or a value between 1~10
-- particle_size: particle size of target molecular
function load_Particle_From_PrePick_t7(inputDIR, inputFiles, trainNumber, particle_size, bin_scale)
    local positive_data = {}
    local negative_data = {}
    if not string.match(inputFiles,"%.t7$") then error("wrong format of input file, must be `.t7` for symbol 3:",inputFiles) end	
    local autopick_file_coordinate=torch.load(inputFiles)
    print("Iteration input file number:",#autopick_file_coordinate)
    local number_high_score=0
    local autopick_total={}
    for i=1,#autopick_file_coordinate do
        for j=1,#autopick_file_coordinate[i] do
            table.insert(autopick_total,autopick_file_coordinate[i][j])
            if autopick_file_coordinate[i][j][3]>0.95 then
                number_high_score = number_high_score+1
            end 
        end
    end
    print("number of particles with high score:",number_high_score)
    print("number of particles total pick(threshold 0):",#autopick_total)	
    --scale the micrograph
    local mrc_filename=paths.concat(inputDIR,autopick_file_coordinate[1][1][4])
    --print(mrc_filename)
    local data = readmrc(mrc_filename)
    local col = data:size(2)
    local row = data:size(3)
    -- need to be manually set
    local scale
    local scale_model 
    if opt.bin then
        scale = bin_scale
        scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
        scale_model.weight:fill(1)
        scale_model.bias:fill(0)
        particle_size = math.ceil(particle_size/scale)
    else
        scale = 1
    end
    -- define the gaussian lowpass filter
    local gaussian_kernel
    if opt.gaussianBlur then
        local kernel_size = tonumber(opt.gaussianKernelSize)
        if math.fmod(kernel_size,2) == 0 then kernel_size = kernel_size+1 end
        gaussian_kernel = image.gaussian(kernel_size, opt.gaussianSigma, 1, true)
    end

    table.sort(autopick_total,function(a,b) return a[3]>b[3] end)
    if trainNumber <= 10 and trainNumber>=1 then
        trainNumber = math.ceil(trainNumber*#autopick_total/10)
    end
    local coordinate_top = get_head(autopick_total,trainNumber)
    local mrc_coordinate = process_coordinate(coordinate_top)
    -- extract the positive and negative data in table
    local p_data={}
    local n_data={}
    local positive_number=0
    local negative_number=0

    for i=1,#mrc_coordinate do
        local filename = mrc_coordinate[i][1][4]
        local mrc_filename=paths.concat(inputDIR,filename)
        local m,n=string.find(filename,"%.mrc$")
        local base=paths.concat(debugDir,string.sub(filename,1,m-1))
        xlua.progress(i,#mrc_coordinate)
        local data = readmrc(mrc_filename)
        --scale the input mrc to reduce
        if opt.bin then
            data = scale_model:forward(data)
        end
        -- do a gaussian lowpass
        if opt.gaussianBlur then
            data = image.convolve(data,gaussian_kernel,'full')
        end

        local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)
        local coordinate_data = mrc_coordinate[i]

        --scale the coordinate
        if opt.bin then
            for j=1,#coordinate_data do
                coordinate_data[j][1] = math.ceil(coordinate_data[j][1]/scale)
                coordinate_data[j][2] = math.ceil(coordinate_data[j][2]/scale)
            end
        end
        local positive_particle_data = pickout_particle(data,coordinate_data,particle_size)
        if positive_particle_data then
            local negative_particle_data,neg_coordinate = pickout_negative_particle(data,coordinate_data,particle_size)
            local filename_particle=base.."_train_particle.jpg"
            --print(filename_particle)
            display_compare(data,coordinate_data,neg_coordinate,particle_size,filename_particle)
            for j=1,negative_particle_data:size(1) do
                table.insert(negative_data,negative_particle_data[j])
            end
            for j=1,positive_particle_data:size(1) do
                table.insert(positive_data,positive_particle_data[j])
            end
            positive_number = positive_number+positive_particle_data:size(1)
            negative_number = negative_number+negative_particle_data:size(1)
        end
        collectgarbage()
    end
    return  positive_data, negative_data
end

-- Load samples from the mrc file directory. 
-- This manner is only used to train a model just using one molecular.
function load_Particle_From_mrcFile_Dir(inputDIR, trainNumber, particle_size, bin_scale, coordinateType, coordinateSymbol, trainMrcNumber)
    mrc_file_all = {}
    mrc_file_label = {}
    coordinate_file = {}
    if not paths.dirp(inputDIR) then
        error("Invalid inputDIR:"..inputDIR)    
    end
    for file in paths.files(inputDIR) do
        if string.match(file,"$%.mrc") then
            mrc_file = paths.concat(inputDIR,file)
            table.insert(mrc_file_all,mrc_file)
         end
    end
    table.sort(mrc_file_all)
    for i=1,#mrc_file_all do
        local mrc_file = mrc_file_all[i]
        local basename = paths.basename(mrc_file)
        local dirname = paths.dirname(mrc_file)
        local j,k = string.find(mrc_file,"%.mrc")
        local name_prex = string.sub(mrc_file,1,j-1)
        if coordinateType == 'eman' then
            local coordinate = name_prex..coordinateSymbol..'.box'
            if paths.filep(coordinate) then
                table.insert(mrc_file_label,mrc_file)
                table.insert(coordinate_file,coordinate)
            end
        elseif coordinateType == 'relion' then
            local coordinate = name_prex..coordinateSymbol..'.star'
            if paths.filep(coordinate) then
                table.insert(mrc_file_label,mrc_file)
                table.insert(coordinate_file,coordinate)
            end
        else
            error('Wrong type: '..coordinateType)
        end
    end
    
    print("Total mrc file:",#mrc_file_all)
    print("Manual pick mrc file:",#mrc_file_label)

    -- manually set the value, the default is 3
    local scale
    local scale_model
    if opt.bin then
        scale = bin_scale
        scale_model = nn.SpatialSubSampling(1,scale,scale,scale,scale)
        scale_model.weight:fill(1)
        scale_model.bias:fill(0)
        particle_size = math.ceil(particle_size/scale)
    else
        scale = 1
    end
    -- define the gaussian lowpass filter
    local gaussian_kernel
    if opt.gaussianBlur then
        local kernel_size = tonumber(opt.gaussianKernelSize)
        if math.fmod(kernel_size,2) == 0 then kernel_size = kernel_size+1 end
        gaussian_kernel = image.gaussian(kernel_size, opt.gaussianSigma, 1, true)
    end

    --extract the positive and negative data in table               
    local positive_data = {}
    local negative_data = {}
    local positive_number = 0
    local negative_number = 0
    if trainMrcNumber == 0 then trainMrcNumber = #mrc_file_label
    else trainMrcNumber = tonumber(trainMrcNumber) end
    for i=1, trainMrcNumber do
        xlua.progress(i, trainMrcNumber)
        local data = readmrc(mrc_file_label[i])

        -- preprocess to micrograph
        -- do the bin preprocess
        if opt.bin then 
            data = scale_model:forward(data)
        end
        -- do a gaussian lowpass
        if opt.gaussianBlur then
            data = image.convolve(data,gaussian_kernel,'full')
        end
        local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)

        local coordinate_data
        if coordinateType == 'eman' then
            coordinate_data = read_coordinate_eman(coordinate_file[i])
        elseif coordinateType == 'relion' then
            coordinate_data = read_coordinate_relion(coordinate_file[i])
        else
            error('Wrong coordinateType: '..coordinateType)
        end

        --scale the coordinate
        if opt.bin then
            for i=1,#coordinate_data do
                coordinate_data[i][1] = math.ceil(coordinate_data[i][1]/scale)
                coordinate_data[i][2] = math.ceil(coordinate_data[i][2]/scale)
            end
        end

        local positive_particle_data = pickout_particle(data, coordinate_data, particle_size)
        if positive_particle_data then
            local negative_particle_data, neg_coordinate = pickout_negative_particle(data, coordinate_data, particle_size)
            for j=1, positive_particle_data:size(1) do
                table.insert(positive_data, positive_particle_data[j])
            end
            for j=1, negative_particle_data:size(1) do
                table.insert(negative_data, negative_particle_data[j])
            end
            positive_number = positive_number+positive_particle_data:size(1)
            negative_number = negative_number+negative_particle_data:size(1)
        end
        collectgarbage()
    end
    print("positive_number:",positive_number)
    print("negative_number:",negative_number)
    return positive_data, negative_data
end
--                

function divide_Particle_Into_Train_And_Evaluation(positive_data, negative_data)
    local positive_number = #positive_data
    local negative_number = #negative_data

    local train_positive = math.ceil(positive_number*0.9)
    local train_negative = math.ceil(negative_number*0.9)

    local test_positive = positive_number-train_positive
    local test_negative = negative_number-train_negative

    local trsize = train_positive+train_negative
    local tesize = test_positive+test_negative

    print('positive_number:',positive_number)
    print('negative_number:',negative_number)
    print('trsize:',trsize)
    print('tesize:',tesize)

    local trainData = {}
    trainData.data = torch.Tensor(trsize,scale_size[1],scale_size[2],scale_size[3])
    trainData.labels = torch.Tensor(trsize)
    trainData.size = trsize 

    local testData = {}
    testData.data = torch.Tensor(tesize,scale_size[1],scale_size[2],scale_size[3])
    testData.labels = torch.Tensor(tesize)
    testData.size = tesize
	
    local posIndices = torch.randperm(positive_number)
    local negIndices = torch.randperm(negative_number)	
    local pos = 1
    local neg = 1
    for i=1,trsize do
        if(i<=train_positive) then
            trainData.data[i] = image.scale(positive_data[ posIndices[pos] ],scale_size[2],scale_size[3])
            trainData.labels[i] = 1
            pos = pos+1
        else
            trainData.data[i] = image.scale(negative_data[ negIndices[neg] ],scale_size[2],scale_size[3])
            trainData.labels[i] = 2
            neg = neg+1
        end
        local mean = trainData.data[i]:mean()
        local std = trainData.data[i]:std()
        trainData.data[i]:add(-mean)
        trainData.data[i]:div(std)
    end

    for i=1,tesize do
        if(i<=test_positive) then
            testData.data[i] = image.scale(positive_data[ posIndices[pos] ],scale_size[2],scale_size[3])
            testData.labels[i] = 1
            pos = pos+1
        else
            testData.data[i] = image.scale(negative_data[ negIndices[neg] ],scale_size[2],scale_size[3])
            testData.labels[i] = 2
            neg = neg+1
        end
        local mean = testData.data[i]:mean()
        local std = testData.data[i]:std()
        testData.data[i]:add(-mean)
        testData.data[i]:div(std)
    end
    local data={}
    table.insert(data,trainData)
    table.insert(data,testData)
    return data
end	

-- extract the positive and negative samples for training
function load_TrainData_From_Relion_Star(inputFiles, trainNumber, particle_size, bin_scale)
    local positive_data, negative_data = load_Particle_From_Relion_Star(inputFiles, trainNumber, particle_size, bin_scale)
    data = divide_Particle_Into_Train_And_Evaluation(positive_data, negative_data)
    return data
end

function load_TrainData_From_Torch_t7(inputDIR, inputFiles, trainNumber, particle_size)
    local positive_data, negative_data = load_Particle_From_Torch_t7(inputDIR, inputFiles, trainNumber, particle_size)
    data = divide_Particle_Into_Train_And_Evaluation(positive_data, negative_data)
    return data
end

function load_TrainData_From_prePick_t7(inputDIR, inputFiles, trainNumber, particle_size, bin_scale)
    local positive_data, negative_data = load_Particle_From_prePick_t7(inputDIR, inputFiles, trainNumber, particle_size, bin_scale)
    data = divide_Particle_Into_Train_And_Evaluation(positive_data, negative_data)
    return data
end

function load_TrainData_From_mrcFile_Dir(inputDIR, trainNumber, particle_size, bin_scale, coordinateType, coordinateSymbol, trainMrcNumber)
    local positive_data, negative_data = load_Particle_From_mrcFile_Dir(inputDIR, trainNumber, particle_size, bin_scale, coordinateType, coordinateSymbol, trainMrcNumber)
    data = divide_Particle_Into_Train_And_Evaluation(positive_data, negative_data)
    return data
end

-- load the evaluation data from the '.t7' format
function loadEvaluationData(input)
	local protein_data_file = {}
	if paths.dirp(input) then
		for file in paths.files(input) do
			if string.match(file,'%.t7$') then
				local filename = paths.concat(input,file)
				table.insert(protein_data_file,filename)
			end
		end		
	elseif paths.filep(input) then
		if string.match(input,'%.t7$') then
			table.insert(protein_data_file,input)
		end
	else
		error("Wrong input!")
	end

	print '==> load evaluation data'
	local positive_data = {}
	local negative_data = {}
	for i=1,#protein_data_file do	
		local protein_data = torch.load(protein_data_file[i])
		for j=1,protein_data[1]:size(1) do
			table.insert(positive_data,protein_data[1][j])
		end
		for j=1,protein_data[2]:size(1) do
			table.insert(negative_data,protein_data[2][j])
		end
	end

	local positive_number = #positive_data
	local negative_number = #negative_data
	local vasize = positive_number+negative_number
	
	local evaluationData = {}
	evaluationData.data = torch.Tensor(vasize,scale_size[1],scale_size[2],scale_size[3])
	evaluationData.labels = torch.Tensor(vasize)
	evaluationData.size = vasize
	
	local posIndices = torch.randperm(positive_number)
	local negIndices = torch.randperm(negative_number)	
		
	local pos = 1
	local neg = 1
	for i=1,vasize do
		if(i<=positive_number) then
                	evaluationData.data[i] = image.scale(positive_data[ posIndices[pos] ],scale_size[2],scale_size[3])
               		evaluationData.labels[i] = 1
               		pos = pos+1
       		else
                	evaluationData.data[i] = image.scale(negative_data[ negIndices[neg] ],scale_size[2],scale_size[3])
             		evaluationData.labels[i] = 2
             		neg = neg+1
        	end
		local mean = evaluationData.data[i]:mean()
		local std = evaluationData.data[i]:std()
		evaluationData.data[i]:add(-mean)
		evaluationData.data[i]:div(std)
	end
	return evaluationData
end
