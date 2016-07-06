----------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Particle picking loaddata')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-input_mrc_dir', 'none', 'the dir of mrc files or image files')
   cmd:option('-particle_size', 100, 'particle size')
   cmd:option('-coordinate_type', 'relion', 'coordinate file formate relion|sim|eman')
   cmd:option('-coordinate_symbol', '', 'coordinate file symbol, like manual , autopick ,or others')
   cmd:option('-extract_mrc_number', -1, 'just for test mrc file number used to train')
   -- preprocess to the micrograph
   cmd:option('-bin', false, 'whether to do the bin preprocess')
   cmd:option('-bin_scale', 3, 'do a bin preprocess to the micrograph image')
   cmd:option('-gaussianBlur', false, 'whether to do gaussian lowpass')
   cmd:option('-gaussianSigma', 0.1, 'define the sigma of the Gaussian kernel')
   cmd:option('-gaussianKernelSize', 5, 'define the size of the Gaussian kernel')
   -- output
   cmd:option('-save_dir', 'relion', 'store the outcome used for debug')
   cmd:option('-save_filename','none', 'the name of saved file of particles(both positive and negative), like "trpv1.t7"')
   cmd:text()
   opt = cmd:parse(arg or {})
   os.execute('mkdir -p '..opt.save_dir)
end
require '1_datafunctions'

function get_mrc_files_from_dir(input_mrc_dir)
    local mrc_file_all = {}
    for file in paths.files(input_mrc_dir) do
        if string.match(file,"%w+%.mrc$") then
            local mrc_file = paths.concat(input_mrc_dir,file)
            table.insert(mrc_file_all,mrc_file)
        end
    end
    return mrc_file_all
end

function extractData(input_mrc_dir, coordinate_type, coordinate_symbol, particle_size, extract_mrc_number, save_filename, save_dir)
    -- get the mrc files with coordinates
    local mrc_file_all = {}
    local mrc_file_label = {}
    local coordinate_file = {}
    if not paths.dirp(input_mrc_dir) then error("Invalid input_mrc_dir:",input_mrc_dir) end
    mrc_file_all = get_mrc_files_from_dir(input_mrc_dir)
    table.sort(mrc_file_all)
    
    for i=1,#mrc_file_all do
        local mrc_file = mrc_file_all[i]
        local basename = paths.basename(mrc_file)
        local dirname = paths.dirname(mrc_file)
        local j,k = string.find(mrc_file,"%.mrc$")
        local name_prex = string.sub(mrc_file,1,j-1)
        if coordinate_type == 'eman' then 
            local coordinate = name_prex..coordinate_symbol..'.box'
            if paths.filep(coordinate) then 
                table.insert(mrc_file_label,mrc_file)
                table.insert(coordinate_file,coordinate)
            end
        elseif coordinate_type == 'relion' then 
            local coordinate = name_prex..coordinate_symbol..'.star'
            if paths.filep(coordinate) then 
                table.insert(mrc_file_label,mrc_file)
                table.insert(coordinate_file,coordinate)
            end
        else
            error('Wrong type '..coordinate_type..'of coordinate type!')
        end
    end
    local mrc_file_number = 0
    if extract_mrc_number <=0 then
        mrc_file_number = #mrc_file_label
    else 
        mrc_file_number = extract_mrc_number
    end
    if mrc_file_number>#mrc_file_label then mrc_file_number = #mrc_file_label end
    print("Total number of mrc file:",#mrc_file_all)
    print("Manually picked number of mrc file:",#mrc_file_label)
    print("Actually used number of mrc file:",mrc_file_number)

    -- manually set the value, the default is 3
    local scale
    local scale_model
    if opt.bin then
        scale = opt.bin_scale
        scale_model = nn.SpatialSubSampling(1, scale, scale, scale, scale)
        scale_model.weight:fill(1)
        scale_model.bias:fill(0)
        particle_size = math.ceil(particle_size/scale)
    else
        scale = 1
    end

    -- preprocess
    -- define the gaussian lowpass filter
    local kernel_size = opt.gaussianKernelSize
    if math.fmod(kernel_size,2) == 0 then kernel_size = kernel_size+1 end
    local gaussian_kernel = image.gaussian(kernel_size, opt.gaussianSigma, 1, true)
	
    --extract the positive and negative data in table		
    local positive_data = {}
    local negative_data = {}
    local positive_number = 0
    local negative_number = 0

    for i=1,mrc_file_number do
        xlua.progress(i,mrc_file_number)
        local data = readmrc(mrc_file_label[i])
        --scale the input mrc to reduce
        data = scale_model:forward(data)
        if opt.gaussianBlur then
            data = image.convolve(data, gaussian_kernel, 'full')
        end
        local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)
	
        local coordinate_data
        if coordinate_type == 'eman' then
            coordinate_data = read_coordinate_eman(coordinate_file[i])
        elseif coordinate_type == 'relion' then
            coordinate_data = read_coordinate_relion(coordinate_file[i])
        else
            error('Wrong type '..coordinate_type..'of coordinate type!')
        end
	
        --scale the coordinate
        if opt.bin then 
            for i=1,#coordinate_data do
                coordinate_data[i][1] = math.ceil(coordinate_data[i][1]/scale)
                coordinate_data[i][2] = math.ceil(coordinate_data[i][2]/scale)
            end
        end
        local positive_particle_data = pickout_particle(data,coordinate_data,particle_size)
        if positive_particle_data then	
            local negative_particle_data, neg_coordinate = pickout_negative_particle(data,coordinate_data,particle_size)
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
    -- save data in tensor
    local saveParticleFilename = paths.concat(save_dir, save_filename)
    local save_data = {}
    table.insert(save_data, positive_data)
    table.insert(save_data, negative_data)
    torch.save(saveParticleFilename,save_data)	
    print('==> save data in '..saveParticleFilename)
   	
    print '==> visualizing data'
    local positive_visualize = torch.Tensor(100, positive_data[1]:size(1), positive_data[1]:size(2), positive_data[1]:size(3))
    local negative_visualize = torch.Tensor(100, positive_data[1]:size(1), positive_data[1]:size(2), positive_data[1]:size(3))
    for i=1, 100 do
        positive_visualize[{{i}}] = positive_data[i]   
        negative_visualize[{{i}}] = negative_data[i]   
    end
    local positive = image.toDisplayTensor{input=positive_visualize, padding = 3, nrow = 10}
    local p_filename = paths.concat(save_dir, save_filename)
    p_filename = p_filename..'_positive.jpg'
    image.save(p_filename, positive)
    local negative = image.toDisplayTensor{input=negative_visualize, padding = 3, nrow = 10}
    local n_filename = paths.concat(save_dir, save_filename)
    n_filename = n_filename..'_negative.jpg'
    image.save(n_filename, negative)
end

extractData(opt.input_mrc_dir, opt.coordinate_type, opt.coordinate_symbol, opt.particle_size, opt.extract_mrc_number, opt.save_filename, opt.save_dir)
