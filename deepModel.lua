require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

deepModel={}
-- define the CNN model structure
function deepModel:init(nfeats, width, height, noutputs, classes) 
      print '==> construct model'
      local nstates = {8,16,32,64}
      local poolsize = 2
      self.noutputs=noutputs
      self.classes=classes
      -- a typical modern convolution network (conv+relu+pool)
      self.model = nn.Sequential()
      if opt.lcn then
	   self.model:add(nn.SpatialSubtractiveNormalization(1,image.gaussian(opt.lcn_size)))
      end 
      local size1=width
      local size2=height
      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      self.model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], 9, 9)) --64-9+1 = 56
      self.model:add(nn.ReLU())
      self.model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) --56/2 = 28

      size1=math.floor((size1-9+1)/2)
      size2=math.floor((size2-9+1)/2)
      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      self.model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], 5, 5)) --28-5+1 = 24
      self.model:add(nn.ReLU())
      self.model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 24/2 = 12

      size1=math.floor((size1-5+1)/2)
      size2=math.floor((size2-5+1)/2)
      -- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
      self.model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], 3, 3)) --12-3+1 = 10
      self.model:add(nn.ReLU())
      self.model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 10/2 = 5

      size1=math.floor((size1-3+1)/2)
      size2=math.floor((size2-3+1)/2)
      -- stage 4 : filter bank -> squashing -> L2 pooling -> normalization
      self.model:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], 2, 2)) --5-2+1 = 4
      self.model:add(nn.ReLU())
      self.model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 4/2 = 2

      size1=math.floor((size1-2+1)/2)
      size2=math.floor((size2-2+1)/2)
      -- stage 5 : standard 2-layer neural network
      
      self.model:add(nn.View(nstates[4]*size1*size2)) -- 64x2x2
      self.model:add(nn.Dropout(0.5))
      self.model:add(nn.Linear(nstates[4]*size1*size2, nstates[4])) --64x2x2 ==> 256
      self.model:add(nn.ReLU())
      self.model:add(nn.Linear(nstates[4], noutputs)) -- 128 ==> 10
      self.model:add(nn.LogSoftMax())
      print '==> here is the model:'
      print(self.model)

      -- define criterion
      self.criterion=nn.ClassNLLCriterion()

      -- define optimizer
      self.optimState = {
           learningRate = opt.learningRate,
           weightDecay = opt.weightDecay,
           momentum = opt.momentum,
           learningRateDecay = 1e-7
      }
      self.optimMethod = optim.sgd
end

-- visualize the feature about the trained model.
function deepModel:display_feature()
	local w_s1=self.model:get(1).weight:clone()
	w_s1:float()
	print(type(w_s1))
	print(#w_s1)
	local w_s2=self.model:get(4).weight:clone()
	w_s2:float()
	print(type(w_s2))
	print(#w_s2)
	local w_s3=self.model:get(7).weight:clone()
	w_s3:float()
	print(type(w_s3))
	print(#w_s3)
	local w_s4=self.model:get(10).weight:clone()
	w_s4:float()
	print(type(w_s4))
	print(#w_s4)
	local filename_s1 = paths.concat(opt.debugdir,"weight_s1.jpg")
	local filename_s2 = paths.concat(opt.debugdir,"weight_s2.jpg")
	local filename_s3 = paths.concat(opt.debugdir,"weight_s3.jpg")
	local filename_s4 = paths.concat(opt.debugdir,"weight_s4.jpg")

	local positive = image.toDisplayTensor{input=w_s1:resize(w_s1:size(1),5,5),padding = 1,nrow = 2}
	image.save(filename_s1,positive)
	local positive = image.toDisplayTensor{input=w_s2:resize(128,5,5),padding = 1,nrow = 8}
	image.save(filename_s2,positive)
	local positive = image.toDisplayTensor{input=w_s3:resize(512,3,3),padding = 1,nrow = 16}
	image.save(filename_s3,positive)
	local positive = image.toDisplayTensor{input=w_s4:resize(2048,2,2),padding = 1,nrow = 32}
	image.save(filename_s4,positive)
end

-- train the model
function deepModel:train(trainData, epoch, trainLogger)
	local time = sys.clock()
	-- Retrieve parameters and gradients:
	-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
	local parameters,gradParameters = self.model:getParameters()
	self.model:training()
	local confusion = optim.ConfusionMatrix(self.classes) -- define the confusionMatrix
	local shuffle=torch.randperm(trainData.size)
	local epoch=epoch or 1
	print('==> doing epoch on training data:')
        print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t=1,trainData.size,opt.batchSize do
		if opt.progressBar then xlua.progress(t,trainData.size) end
		-- creat mini-batch
		local inputs={}
		local targets={}
		for i = t,math.min(t+opt.batchSize-1,trainData.size) do
         		-- load new sample
			local input
         		if opt.rotate then input = jitter(trainData.data[shuffle[i]])
         		else input=trainData.data[shuffle[i]] end

         		local target = trainData.labels[shuffle[i]]

         		if opt.type == 'double' then input = input:double()
         		elseif opt.type == 'cuda' then input = input:cuda()
         		elseif opt.type == 'float' then input = input:float() end
         		table.insert(inputs, input)
         		table.insert(targets, target)
      		end	
		
		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then parameters:copy(x) end
			-- reset gradients 
			gradParameters:zero()
			-- f is the average of all criterions
			local f = 0
			-- evaluate function for complete mini-batch
			for i=1,#inputs do
				-- estimate f
				local output = self.model:forward(inputs[i])
				if opt.type == 'cuda' then output=output:float() end
				local err = self.criterion:forward(output,targets[i])
				f = f + err
			
				-- estimate df/dW
				local df_do = self.criterion:backward(output,targets[i])
				if opt.type == 'cuda' then df_do = df_do:cuda() end
				self.model:backward(inputs[i],df_do)

				-- update confusion
				confusion:add(output,targets[i])
			end
			gradParameters:div(#inputs)
			f = f/#inputs
			return f,gradParameters			
	       end

	       -- optimize on current mini-batch
	       self.optimMethod(feval,parameters,self.optimState)
	end
	time = sys.clock() - time
	time = time/trainData.size
	print("\n ==> time to learn 1 sample = ".. (time*1000) .. 'ms')
	-- print confusion matrix
	print(confusion)	
 
	-- update logger/plot
   	local train_accuracy = confusion.totalValid
   	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	return train_accuracy
end

-- evaluation 
function deepModel:evaluation(testData, testLogger)
	local time = sys.clock()
	-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
	self.model:evaluate()
	
	local confusion = optim.ConfusionMatrix(self.classes) -- define the confusionMatrix
	-- test over test data
	print('==> testing on the test set:')
	for t=1,testData.size do
		if opt.progressBar then xlua.progress(t,testData.size) end
		-- get new sample
                if opt.rotate then
		    local input = testData.data[t]
                    input = jitter_evaluation(input)
		    if opt.type == 'double' then input = input:double()
		    elseif opt.type == 'cuda' then input = input:cuda()
		    elseif opt.type == 'float' then input = input:float() end
		    local target = testData.labels[t]
		    local pred = self.model:forward(input)
                    pred = pred:float()
                    pred:exp()
	            pred = pred:mean(1)[1]
	            pred:div(pred:sum())
	            pred:log()  
		    confusion:add(pred,target)
                else
		    local input = testData.data[t]
		    if opt.type == 'double' then input = input:double()
		    elseif opt.type == 'cuda' then input = input:cuda()
		    elseif opt.type == 'float' then input = input:float() end
		    local target = testData.labels[t]
		    -- test sample
		    local pred = self.model:forward(input)
		    confusion:add(pred,target)
                end
	end
	time = sys.clock() - time
	time = time/testData.size
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
	
	--print confusion matrix
	print(confusion)
	local test_accuracy = confusion.totalValid
	-- update log
	testLogger:add{['% mean class accuracy (test set)'] = test_accuracy * 100}
	return test_accuracy
end

function deepModel:prediction(predictData, batch_size, opt)
    self.model:evaluate()
    prediction = torch.Tensor(predictData.data:size(1), 2)
    for t=1, predictData.size, batch_size do
        -- creat mini-batch
        batch_end_index = math.min(t+batch_size-1, predictData.size) 
        size = batch_end_index-t+1
        batch_data_input = predictData.data:narrow(1, t, size)
        print(batch_data_input:size(1))  
        if opt.type == 'double' then batch_data_input = batch_data_input:double()
        elseif opt.type == 'cuda' then batch_data_input = batch_data_input:cuda()
        elseif opt.type == 'float' then batch_data_input = batch_data_input:float() end
        pred = self.model:forward(batch_data_input) 
        pred:float()
        pred:exp()
        for i=1, pred:size(1) do
            prediction[t+i-1][1] = pred[i][1]
            prediction[t+i-1][2] = pred[i][2]
        end
        collectgarbage()
    end
    return prediction
end
