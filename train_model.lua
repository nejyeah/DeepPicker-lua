----------------------------------------------------------------------
require 'torch'
require 'image'
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Particle picking, train model')
cmd:text()
cmd:text('Options:')
-- about the model 
cmd:option('-seed', 1, 'fixed input random seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads,implemented by the torch itself')
-- loading data 
cmd:option('-withoutTrain', false, 'whether to retrain when load a model from local')
cmd:option('-retrain', 'none', 'retrain already existed model')
cmd:option('-trainType', 1, 'select which format of input data, type 1|2|3|4, 1 for single molecule, 2 for multiple molecules, 3 for iterative training, 4 for cooperating with Relion 2D classification')
cmd:option('-inputDIR', './cache', 'the dir of mrc files or others depends on the trainType')
cmd:option('-inputFiles', '', "starfiles or t7 file depends on the trainType ")
cmd:option('-trainNumber', 0, 'number of positive particles used to train classifier')
cmd:option('-trainMrcNumber', 0, 'number of mrc files to be trained')
cmd:option('-particle_size', 180, 'the particle size of the training data')
cmd:option('-coordinateType', 'relion', 'the format of the coordinate, relion|eman')
cmd:option('-coordinateSymbol', '', 'if the mrc file name is stack_001.mrc, and the coordinate file is stack_001_manual.star, then the symbol is _manual')
cmd:option('-scale_size', 64, 'all the particles are scaled to this size, and this is also the input patch size of the model.')

-- preprocess
-- to micrograph
cmd:option('-bin', false, 'whether to do a bin process')
cmd:option('-bin_scale', 1, 'do a bin preprocess to the micrograph before extracting the particles')
cmd:option('-gaussianBlur', false, 'whether to do gaussian lowpass')
cmd:option('-gaussianSigma', 0.1, 'define the sigma of the Gaussian kernel')
cmd:option('-gaussianKernelSize', 5, 'define the size of the Gaussian kernel')
cmd:option('-histogram_equalization', false, 'whether to do the preprocess')
cmd:option('-edgeDetect', false, 'whether to do the edge detction for choosing false particle.')
-- to particle
cmd:option('-rotate', false, ' whether to do the random rotation')
cmd:option('-lcn', false, 'whether to do LCN with the input small patch')
cmd:option('-lcn_size', 9, 'local contrast size')

-- training
cmd:option('-model_save_dir', 'results', 'subdirectory to save/log experiments in')
cmd:option('-model_symbol', 'symbol', 'if the value is demo, then the model will save in file model_demo.net')
cmd:option('-learningRate', 1e-2, 'learning rate, better not change')
cmd:option('-batchSize', 200, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-progressBar', true, 'whether to use progress Bar')
cmd:option('-debugDir', 'train_samples', 'dir used to store some debug information')
-- evaluation
cmd:option('-evaluation', 'none', 'the data used to evaluation, is a t7 file')
-- visualize
cmd:option('-visualize', false, 'whether to use QT tools to visualize some pictures results')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
----------------------------------------------------------------------
print '==> executing all'
os.execute('mkdir -p '..opt.debugDir)

-- define some parameters 
scale_size={1, opt.scale_size, opt.scale_size}       -- scale size as the input to the model
local noutputs = 2         -- number of classes
local classes = {'positive','negative'}  -- symbol of different classes

if opt.visualize then require '1_datafunctions_qt' end
require '1_data'
require 'deepModel'
-- initialize the model
deepModel:init(scale_size[1], scale_size[2], scale_size[3], noutputs, classes)

-- load the existed model to intialize the parameters
if opt.retrain ~='none' then
    local parameters,gradParameters = deepModel.model:getParameters()
    modelExist = torch.load(opt.retrain)
    local mod2 = modelExist:float()
    local p2,gp2 = mod2:getParameters()
    parameters:copy(p2)
    gradParameters:copy(gp2)
end
if opt.type=="cuda" then deepModel.model:cuda() end

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.model_save_dir, 'train_'..opt.model_symbol..'.log'))
local testLogger = optim.Logger(paths.concat(opt.model_save_dir, 'test_'..opt.model_symbol..'.log'))

-------------------------------------------------------

-- train 
if not opt.withoutTrain then 
    print '==> training'
    --define some parameters of early-stopping
    local n_epochs = 200
    local best_validation_accuracy = 0
    local done_looping = false

    local toleration = 10
    local descend_toleration = 1
    local tolerationflag = 1

    local epoch = 1
    local time = sys.clock()
    local data
    if opt.trainType == 1 then
        data = load_TrainData_From_mrcFile_Dir(opt.inputDIR, tonumber(opt.tranNumber), tonumber(opt.particle_size), tonumber(opt.bin_scale), opt.coordinateType, opt.coordinateSymbol, tonumber(opt.trainMrcNumber))
    elseif opt.trainType == 2 then
        data = load_TrainData_From_Torch_t7(opt.inputDIR, opt.inputFiles, tonumber(opt.trainNumber), tonumber(opt.particle_size))
    elseif opt.trainType == 3 then
        data = load_TrainData_From_prePick_t7(opt.inputDIR, opt.inputFiles, tonumber(opt.trainNumber), tonumber(opt.particle_size), tonumber(opt.bin_scale))
    elseif opt.trainType == 4 then
        data = load_TrainData_From_Relion_Star(opt.inputFiles, tonumber(opt.trainNumber), tonumber(opt.particle_size), tonumber(opt.bin_scale))

    else
        error("Invalid trainType:", opt.trainType)
    end

    local trainData=data[1]
    local testData=data[2]
    while epoch<n_epochs and not done_looping do
        local r_accuracy = deepModel:train(trainData,epoch,trainLogger)
        collectgarbage()
        local validation_accuracy = deepModel:evaluation(testData,testLogger)
        collectgarbage()
        if validation_accuracy > best_validation_accuracy then
            tolerationflag =1
            descend_toleration =1
            best_validation_accuracy = validation_accuracy
        else
            tolerationflag = tolerationflag+1
            descend_toleration = descend_toleration +1
        end

        if tolerationflag> toleration  then done_looping = true end
        if descend_toleration==5 then
            deepModel.optimState.learningRate = deepModel.optimState.learningRate/5
            descend_toleration = 1
        end
        epoch = epoch + 1
        collectgarbage()
    end
    local time = sys.clock() - time
    time = time/60
    print('epoch:'..epoch..'\tTrain a curse classifier , all time cost:'..time..' min')

    --save model and some parameters
    local filename = paths.concat(opt.model_save_dir, 'model_'..opt.model_symbol..'.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    if opt.type == 'cuda' then deepModel.model:float() end
    torch.save(filename, deepModel.model)
end

if opt.evaluation ~='none' then
    print '==> evaluation'
    if opt.type == 'cuda' then deepModel.model:cuda() end
    local evaluationData = loadEvaluationData(opt.evaluation)
    local validation_accuracy = deepModel:evaluation(evaluationData,testLogger)
end

