----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'image'
require 'parallel'
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Particle picking')
cmd:text()
cmd:text('Options:')
-- about the model 
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-process', 2, 'number of process')
cmd:option('-iteration', 1, 'times of iteration')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-progressBar', true, 'whether to use progress Bar')

-- autopick 
cmd:option('-trainedModel', 'none', 'trained model')
cmd:option('-input', 'none', 'the dir of mrc files or  mrcfile')
cmd:option('-particle_size', 180, 'particle size')
cmd:option('-coorType', 'relion', 'relion|eman, choose which coordinate formate to output')
cmd:option('-scale_size', 64, 'all the particles are scaled to this size, and this is also the input patch size of the model.')
cmd:option('-visualize', false, 'whether to use QT tools to visualize some results')
cmd:option('-debugDir', 'debug', 'store some results')
cmd:option('-pick_number', 100, 'number of micrographs to pick')
cmd:option('-threshold', 0.5, 'get the particles above the probability threshold or the top number of particles sort by probability(each micrograph)')
cmd:option('-step_size', 4, 'define the step size when the classifier scan the mrc map')
cmd:option('-minDistanceBetweenParticleRate', 0.8, 'set the min Distance of two picked particles, Length = particle_size*minDistanceBetweenParticleRate')

-- preprocess
-- to micrograph
cmd:option('-bin', false, 'whether to do a bin preprocess to the micrograph')
cmd:option('-bin_scale', 3, 'do a bin preprocess to the micrograph.')
cmd:option('-gaussianBlur', false, 'whether to do gaussian lowpass')
cmd:option('-gaussianSigma', 0.1, 'define the sigma of the Gaussian kernel')
cmd:option('-gaussianKernelSize', 5, 'define the sigma of the Gaussian kernel')
cmd:option('-histogram_equalization', false, 'whether to do the preprocess')
-- to particle
cmd:option('-rotate', false, 'whether to do the preprocess')
cmd:option('-lcn', false, 'whether to do LCN with the input small patch')
cmd:option('-lcn_size', 9, 'local contrast size')

-- trick to abolish negative particle
-- preprocess
-- delete negative particle based on edge detection algorithm
cmd:option('-EdgeDetect', false, 'using the method Canny edge detection to delete ice noise')
cmd:option('-particle_edge_notable', false, 'if the value is true, it will keep those edges, the length of which are comparable to particle size')
cmd:option('-carbonFilmDetect', false, 'if the value is true, it will try to detect edge of carbon film during the canny edge detection')
-- post process
-- refine the coordinate
cmd:option('-refineCoordinate', false, 'whether to refine the center of coordinate')
-- delete negative particle based on extrem values
cmd:option('-postProcess', false, 'whether to delete ice based on some properties')
cmd:option('-postProcessStd', 3, 'choose the ratio of the std value')
-- delete negative particle based on connected componnent analysis
cmd:option('-deleteIceByConnectedArea', false, 'delete the large connected area')
cmd:option('-meanRate', 3, 'used to abolish the large connected area above mean*meanRate')
cmd:option('-binaryThreshold', 0.5, 'used to abolish the large connected area above mean*meanRate')
-- delete negative particle based on another specific classifier
cmd:option('-IceClassifier', 'none', 'the specific classifier train based on particle and ice' )
cmd:text()
opt = cmd:parse(arg or {})
local time_start = sys.clock()
-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.gpuid)
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
----------------------------------------------------------------------
-- initialize output dir and file
print '==> executing all'
os.execute('mkdir -p '..opt.debugDir)

-- define some parameters 
scale_size={1, opt.scale_size, opt.scale_size}       -- scale size as the input to the model
local noutputs = 2         -- number of classes
local classes = {'positive', 'negative'}  -- symbol of different classes
----------------------------------------------------------------------
-- initialize input filename and model definition
require 'deepModel'
-- initialize the model using the existed model
deepModel:init(scale_size[1],scale_size[2],scale_size[3],noutputs,classes)
local parameters,gradParameters = deepModel.model:getParameters()
local modelExist = torch.load(opt.trainedModel)
local mod2 = modelExist:float()
local p2,gp2 = mod2:getParameters()
local p2,gp2 = modelExist:getParameters()
parameters:copy(p2)
gradParameters:copy(gp2)
deepModel.model:evaluate()
----------------------------------------------------------------------
-- load the data
if opt.visualize then require '1_datafunctions_qt' end
require '1_datafunctions'
require '1_data'
InputData:loadMrcFiles(opt.input)

local mrc_file_autopick = InputData.mrc_file_all
print("Total number of mrc files:",#mrc_file_autopick)
if opt.pick_number>#mrc_file_autopick then
	print("To be picked number:",#mrc_file_autopick)
else
	print("To be picked number:",opt.pick_number)
end
------------------------------------------------------------------
-- initialize autoPicker
require 'autoPicker'
autoPicker:init(mrc_file_autopick, deepModel, scale_size, opt.bin_scale, opt)

-- do autopick
for i=1,opt.iteration do
	local parallel_autopick=autoPicker.parallel_autopick
	local ok,err = pcall(parallel_autopick,autoPicker)
	if not ok then
		print(err)
		parallel.close()
	end
end

