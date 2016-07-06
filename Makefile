# your mrc files dir
# edit it
autopick_dir=/media/bioserver1/Data/paper_test/trpv1/test/lowpass

# as small as possible 
# edit it
particle_size=180

# type cuda|float|double
# GPU will be used when type is cuda, cuda-tookit is required.
# better no change
type=cuda
# number of process to use when do autopicking
# edit it
process=2

## autopick->iteration_train->autopick
# number of mrc files to be picked
# edit it
picknumber1=2

## train parameter, take the particles with score above the train_threshold to retrain the model
# 0.9 is recommended
# edit it
train_threshold=0.9

# 10 to 200 is recommended based on the trainNumber
# edit it
batchSize=10

# better not to change it 
learningRate=0.01

## autopick parameter
## numbers of mrc files to be picked, can be set to a large number to pick all the mrc files
# edit it
picknumber2=100

## symbol of the picked molecular. You can change it as you want
# edit it
target=trpv1

## symbol of the pretrained model. Do not change it, or you have another trained model.
protein0=demo
trained_model=all_lowpass_demo

iteration_autopick:
     

pick:
	qlua autopick.lua -input /media/bioserver1/Data/paper_test/trpv1/test/lowpass -debugDir autopick-trpv1-by-trpv1-lg10A-s64-10000 -trainedModel ./results/model_trpv1_LG10A_S64.net -particle_size 180 -pick_number 100 -threshold 0.5 -visualize -deleteIceByConnectedArea -refineCoordinate -postProcess -postProcessStd 3 -bin -process 5 

train:
	#qlua train_model.lua -trainType 2 -inputDIR ../data_extract -inputFiles trpv1_train_lowpass_gauss_20A.t7 -trainNumber 10000 -particle_size 180 -scale_size 64 -rotate -model_symbol trpv1_LG20A_S64 	
	#qlua train_model.lua -trainType 2 -inputDIR ../data_extract -inputFiles trpv1_train_lowpass_gauss_10A.t7 -trainNumber 10000 -particle_size 180 -scale_size 100 -rotate -model_symbol trpv1_LG10A_S100 	
	qlua autopick.lua -input /media/bioserver1/Data/paper_test/trpv1/test/lowpass -debugDir autopick-trpv1-by-trpv1-lg10A-s100-10000 -trainedModel ./results/model_trpv1_LG10A_S100.net -particle_size 180 -pick_number 100 -threshold 0.5 -visualize -deleteIceByConnectedArea -refineCoordinate -postProcess -postProcessStd 3 -bin -process 5 -scale_size 100 
	#qlua train_model.lua -trainType 2 -inputDIR ../data_extract -inputFiles trpv1_train_lowpass_tanh_20A.t7 -trainNumber 10000 -particle_size 180 -scale_size 64 -rotate -model_symbol trpv1_LT20A_S64 	

lowpass:
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/gammas/test/original -outputDIR /media/bioserver1/Data/paper_test/gammas/test/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/gammas/train/original -outputDIR /media/bioserver1/Data/paper_test/gammas/train/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/ss/test/original -outputDIR /media/bioserver1/Data/paper_test/ss/test/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/ss/train/original -outputDIR /media/bioserver1/Data/paper_test/ss/train/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/trpv1/test/original -outputDIR /media/bioserver1/Data/paper_test/trpv1/test/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/trpv1/train/original -outputDIR /media/bioserver1/Data/paper_test/trpv1/train/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/snare/train/original -outputDIR /media/bioserver1/Data/paper_test/snare/train/lowpass_tanh_20A -sigma 0.05   	
	th lowpass_mrc_by_EMAN.lua -inputDIR /media/bioserver1/Data/paper_test/beta/train/original -outputDIR /media/bioserver1/Data/paper_test/beta/train/lowpass_tanh_20A -sigma 0.05   	

extract:
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/gammas/train/lowpass_tanh_20A -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'gammas_train_lowpass_tanh_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/gammas/train/lowpass_20A -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'gammas_train_lowpass_gauss_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/gammas/train/lowpass -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'gammas_train_lowpass_gauss_10A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/ss/train/lowpass_tanh_20A -particle_size 320 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'ss_train_lowpass_tanh_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/ss/train/lowpass_20A -particle_size 320 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'ss_train_lowpass_gauss_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/ss/train/lowpass -particle_size 320 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'ss_train_lowpass_gauss_10A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/trpv1/train/lowpass_tanh_20A -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'trpv1_train_lowpass_tanh_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/trpv1/train/lowpass_20A -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'trpv1_train_lowpass_gauss_20A.t7'  
	qlua extractData.lua -input_mrc_dir /media/bioserver1/Data/paper_test/trpv1/train/lowpass -particle_size 180 -coordinate_symbol '_manual_checked' -bin -save_dir '../data_extract' -save_filename 'trpv1_train_lowpass_gauss_10A.t7'  

analysis:
	qlua analysis_coordinate.lua -compare_with_manual -inputPickCoordinate ./autopick-trpv1-by-trpv1-lt20A-s64-10000/coordinate_autopick.t7 -inputDIR /media/bioserver1/Data/paper_test/trpv1/test/lowpass_tanh_20A -particle_size 180 -coordinateSymbol _refine_frealign -display
