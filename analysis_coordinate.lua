------------------------------------------------------------------------------
-- Author WangFeng  2016/04/14
-- This script can be used to analysis the autopick results 'coordinate_autopick.t7'
-- It will analysis the picked coordinate compared with manually picked coordinates
-- Recall and Precision will be estimated
-- Also the PR-plot will be produced

-- Input:
--      -input: the dir of mrc files(coordinate files should be in the same dir)
--      -scale: wether scale the jpg to a small size
--      -particle_size: particle size
--      -coorType1: relion|eman
--      -coorSymbol1: if star file is 'stack_001_corr_manual.star', then the coorSymbol should be '_manual'
--                    if star file is 'stack_001_corr.star', then the coorSymbol should be ''
--      -number: number of mrc files to visualize
-- Output:
--      the images of plot particles will be stored in the same directory of the mrc file
-------------------------------------------------------------------------------

require '1_datafunctions'
require '1_datafunctions_qt'
require 'gnuplot'
-- load data
if not opt then
    print '==> mrc2jpg options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('mrc file or dir to jpg')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-inputPickCoordinate','none','input picked coordinate file, e.g.,PATH/coordinate_autopick.t7')
    -- compare the picking results with the reference
    cmd:option('-compare_with_manual',false, 'whether to compare the pick results')
    cmd:option('-inputDIR','none','the dir of mrc files as well as the coordinate file. Both should be saved on the same dir.')
    cmd:option('-particle_size',180,'particle size(pixel)')
    cmd:option('-coordinateType','relion','the type of coordinate file')
    cmd:option('-coordinateSymbol','','none,means nothing,the symbol of coordinate file')
    cmd:option('-display', false, 'whether to show the results in image')
    cmd:option('-write_distance_all', false, 'whether to write the all distance of particles')
    -- save the picking results in coordinate file for a specific threshold
    cmd:option('-save_coordinate',false, 'whether save the coordinate based on a specific threshold')
    cmd:option('-save_threshold',0.5, 'the threshold to choose particle')
    cmd:option('-save_coordinateType','relion','the type of coordinate file')
    cmd:option('-save_coordinateSymbol','_CNNpick','none,means nothing,the symbol of coordinate file')
    -- save the prediction score, used to analysis the distribution of the prediction score
    cmd:option('-save_prediction_score',false,"Whether to save the prediction score in a single file")
    opt=cmd:parse(arg or {})
end

local inputDIR=opt.inputDIR
local inputFiles=opt.inputPickCoordinate
local coordinate_symbol=opt.coordinateSymbol
local particle_size = opt.particle_size
opt.debugdir=paths.dirname(inputFiles)

if not string.match(inputFiles,"%.t7") then 
    error("wrong format of input file, must be `.t7` for symbol 3:",inputFiles) 
end
	
if opt.compare_with_manual and opt.display then
    if not paths.dirp(inputDIR) then error("inputDIR must be a directory:",inputDIR) end
    -- load results file
    local autopick_file_coordinate=torch.load(inputFiles)
    --scale the micrograph
    local mrc_filename=paths.concat(inputDIR,autopick_file_coordinate[1][1][4])
    -- print(mrc_filename)
    local data = readmrc(mrc_filename)
    local col = data:size(2)
    local row = data:size(3)
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
    local number_high_score=0
    local autopick_total={}
    local total_pick = 0
    local total_manualpick = 0
    local tp = 0
    for i=1,#autopick_file_coordinate do
        local mrc_filename=autopick_file_coordinate[i][1][4]
        -- read manual pick coordinate
        local j,k = string.find(mrc_filename,"%.mrc$")
        local name_prex = string.sub(mrc_filename,1,j-1)
        local coordinate_file
        if opt.coordinateType == 'relion' then
            coordinate_file=paths.concat(inputDIR,name_prex..coordinate_symbol..'.star')
        elseif opt.coordinateType == 'eman' then
            coordinate_file=paths.concat(inputDIR,name_prex..coordinate_symbol..'.box')
        else
            error("Wrong type of coordinateType, it must be relion|eman!")
        end

        -- load mrc file
        mrc_filename = paths.concat(inputDIR,mrc_filename)
        local data = readmrc(mrc_filename)
        --scale the input mrc to reduce
        data = scale_model:forward(data)
        local max = data:max()
        local min = data:min()
        data:add(-min):div(max-min)

        -- load coordinate file
        local coordinate_manual
        if paths.filep(coordinate_file) then
            coordinate_manual = read_coordinate_relion(coordinate_file)
        else
            error("Can not find manual coordinate file:",coordinate_file)
        end	
        total_manualpick = total_manualpick+#coordinate_manual
				
        for j=1,#coordinate_manual do
            coordinate_manual[j][5] = 0
        end
        for j=1,#autopick_file_coordinate[i] do
            autopick_file_coordinate[i][j][5] = 0
        end
        local tp_sigle,average_distance = test_correlation(autopick_file_coordinate[i],coordinate_manual,particle_size*0.2*scale)
        -- scale the manual coordinate	
        for j=1,#coordinate_manual do
            coordinate_manual[j][1] = math.ceil(coordinate_manual[j][1]/scale)
            coordinate_manual[j][2] = math.ceil(coordinate_manual[j][2]/scale)
        end

        -- scale the autopick coordinate
        for j=1,#autopick_file_coordinate[i] do
            autopick_file_coordinate[i][j][1] = math.ceil(autopick_file_coordinate[i][j][1]/scale)
            autopick_file_coordinate[i][j][2] = math.ceil(autopick_file_coordinate[i][j][2]/scale)
        end
	
        local image_name = name_prex..'_score_0.95.jpg'
        image_name = paths.concat(opt.debugdir,image_name)
        display_compare(data,coordinate_manual,autopick_file_coordinate[i],particle_size,image_name,2)		
        -- calculate the number of true positive
        --local total_pick = 0
        local tp_sigle = 0
        for j=1,#autopick_file_coordinate[i] do
            table.insert(autopick_total,autopick_file_coordinate[i][j])
            if autopick_file_coordinate[i][j][3]>0.95 then
                number_high_score = number_high_score+1
            end
            if autopick_file_coordinate[i][j][3]>0.5 then
                total_pick=total_pick+1
                if autopick_file_coordinate[i][j][5] == 1 then
                    tp = tp + 1
                    tp_sigle = tp_sigle + 1
                end
            end
        end

        local recall = tp_sigle/#coordinate_manual
        print(recall)
    end
    local precision=tp/total_pick
    local recall = tp/total_manualpick
    print("(threshold 0.5)precision:"..precision.."\trecall:"..recall)
    print("number of particles with high score(over 0.95):",number_high_score)

    table.sort(autopick_total,function(a,b) return a[3]>b[3] end)
    -- plot some graphs
    local total_tp = {}
    local total_recall = {}
    local total_precision = {}
    local total_probability = {}
    local average_distance = {}
    local tp_all_distance = {}
    local total_distance = 0
    local tem=0 -- current true positive number
    local index= 0
    for i=1,#autopick_total do
        -- compute recall,precision
        index = index+1
        if autopick_total[i][5] == 1 then
            tem = tem+1
            table.insert(tp_all_distance, autopick_total[i][6])
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
        if tem == 0 then
            average = 0
        else
            average = total_distance/tem
        end
        table.insert(average_distance,average)
    end
    -- write the results
    local total_results_file = paths.concat(opt.debugdir,'results.txt')
    local fp_r = torch.DiskFile(total_results_file,"w")
    fp_r:writeString(table.concat(total_tp,' ')..'\n')
    fp_r:writeString(table.concat(total_recall,' ')..'\n')
    fp_r:writeString(table.concat(total_precision,' ')..'\n')
    fp_r:writeString(table.concat(total_probability,' ')..'\n')
    fp_r:writeString(table.concat(average_distance,' ')..'\n')
    fp_r:writeString('#total autopick number:'..#autopick_total..'\n')
    fp_r:writeString('#total manual pick number:'..total_manualpick..'\n')
    fp_r:writeString('#the first row is number of true positive\n')
    fp_r:writeString('#the second row is recall\n')
    fp_r:writeString('#the third row is precision\n')
    fp_r:writeString('#the fourth row is probability\n')
    fp_r:writeString('#the fiveth row is distance\n')
    local timesOfmanual = math.ceil(#autopick_total/total_manualpick)
    for i=1,timesOfmanual do
        print('autopick_total sort, take the head number of total_manualpick * ratio '..i)
        fp_r:writeString('#autopick_total sort, take the head number of total_manualpick * ratio '..i..'\n')
        if i==timesOfmanual then
            print('precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall])
            fp_r:writeString('#precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall]..'\n')
        else
            print('precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i])
            fp_r:writeString('#precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i]..'\n')
        end
    end
    fp_r:close()

    if opt.write_distance_all then
        local total_results_file = paths.concat(opt.debugdir,'results_distance_all.txt')
        local fp_r = torch.DiskFile(total_results_file,"w")
        for i=1, #tp_all_distance do
            fp_r:writeString(tp_all_distance[i]..'\n')
        end
        fp_r:close()
    end

    -- plot the graph
    local plot_length = 4*total_manualpick
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
    local graph_name=paths.concat(opt.debugdir,"autopick.png")
    gnuplot.pngfigure(graph_name)
    gnuplot.plot({'probability',x_value,probability,'-'},{'recall',x_value,recall,'-'},{'precision',x_value,precision,'-'},{'distance',x_value,averageDistance,'-'})		
    gnuplot.xlabel('distance is normalized by 50')
    gnuplot.grid(true)
    gnuplot.title('Recall-Precision-Probability-Distance')
    gnuplot.plotflush()
end

if opt.compare_with_manual and not opt.display then
    if not paths.dirp(inputDIR) then error("inputDIR must be a directory:",inputDIR) end
    -- load results file
    local autopick_file_coordinate=torch.load(inputFiles)
    --scale the micrograph
    local mrc_filename=paths.concat(inputDIR,autopick_file_coordinate[1][1][4])
    local number_high_score=0
    local autopick_total={}
    local total_pick = 0
    local total_manualpick = 0
    local tp = 0
    for i=1,#autopick_file_coordinate do
        local mrc_filename=autopick_file_coordinate[i][1][4]
        -- read manual pick coordinate
        local j,k = string.find(mrc_filename,"%.mrc$")
        local name_prex = string.sub(mrc_filename,1,j-1)
        local coordinate_file
        if opt.coordinateType == 'relion' then
            coordinate_file=paths.concat(inputDIR,name_prex..coordinate_symbol..'.star')
        elseif opt.coordinateType == 'eman' then
            coordinate_file=paths.concat(inputDIR,name_prex..coordinate_symbol..'.box')
        else
            error("Wrong type of coordinateType, it must be relion|eman!")
        end

        -- load coordinate file
        local coordinate_manual
        if paths.filep(coordinate_file) then
            coordinate_manual = read_coordinate_relion(coordinate_file)
        else
            error("Can not find manual coordinate file:",coordinate_file)
        end	
        total_manualpick = total_manualpick+#coordinate_manual
				
        for j=1,#coordinate_manual do
            coordinate_manual[j][5] = 0
        end
        for j=1,#autopick_file_coordinate[i] do
            autopick_file_coordinate[i][j][5] = 0
        end
        local tp_sigle,average_distance = test_correlation(autopick_file_coordinate[i],coordinate_manual,particle_size*0.2)
        local tp_sigle = 0
        for j=1,#autopick_file_coordinate[i] do
            table.insert(autopick_total,autopick_file_coordinate[i][j])
            if autopick_file_coordinate[i][j][3]>0.95 then
                number_high_score = number_high_score+1
            end
            if autopick_file_coordinate[i][j][3]>0.5 then
                total_pick=total_pick+1
                if autopick_file_coordinate[i][j][5] == 1 then
                    tp = tp + 1
                    tp_sigle = tp_sigle + 1
                end
            end
        end

        local recall = tp_sigle/#coordinate_manual
        print(recall)
    end
    local precision=tp/total_pick
    local recall = tp/total_manualpick
    print("(threshold 0.5)precision:"..precision.."\trecall:"..recall)
    print("number of particles with high score(over 0.95):",number_high_score)

    table.sort(autopick_total,function(a,b) return a[3]>b[3] end)
    -- plot some graphs
    local total_tp = {}
    local total_recall = {}
    local total_precision = {}
    local total_probability = {}
    local average_distance = {}
    local tp_all_distance = {}
    local total_distance = 0
    local tem=0 -- current true positive number
    local index= 0
    for i=1,#autopick_total do
        -- compute recall,precision
        index = index+1
        if autopick_total[i][5] == 1 then
            tem = tem+1
            table.insert(tp_all_distance, autopick_total[i][6])
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
        if tem == 0 then
            average = 0
        else
            average = total_distance/tem
        end
        table.insert(average_distance,average)
    end
    -- write the results
    local total_results_file = paths.concat(opt.debugdir,'results.txt')
    local fp_r = torch.DiskFile(total_results_file,"w")
    fp_r:writeString(table.concat(total_tp,' ')..'\n')
    fp_r:writeString(table.concat(total_recall,' ')..'\n')
    fp_r:writeString(table.concat(total_precision,' ')..'\n')
    fp_r:writeString(table.concat(total_probability,' ')..'\n')
    fp_r:writeString(table.concat(average_distance,' ')..'\n')
    fp_r:writeString('#total autopick number:'..#autopick_total..'\n')
    fp_r:writeString('#total manual pick number:'..total_manualpick..'\n')
    fp_r:writeString('#the first row is number of true positive\n')
    fp_r:writeString('#the second row is recall\n')
    fp_r:writeString('#the third row is precision\n')
    fp_r:writeString('#the fourth row is probability\n')
    fp_r:writeString('#the fiveth row is distance\n')
    local timesOfmanual = math.ceil(#autopick_total/total_manualpick)
    for i=1,timesOfmanual do
        print('autopick_total sort, take the head number of total_manualpick * ratio '..i)
        fp_r:writeString('#autopick_total sort, take the head number of total_manualpick * ratio '..i..'\n')
        if i==timesOfmanual then
            print('precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall])
            fp_r:writeString('#precision:'..total_precision[#total_precision]..'\trecall:'..total_recall[#total_recall]..'\n')
        else
            print('precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i])
            fp_r:writeString('#precision:'..total_precision[total_manualpick*i]..'\trecall:'..total_recall[total_manualpick*i]..'\n')
        end
    end
    fp_r:close()

    if opt.write_distance_all then
        local total_results_file = paths.concat(opt.debugdir,'results_distance_all.txt')
        local fp_r = torch.DiskFile(total_results_file,"w")
        for i=1, #tp_all_distance do
            fp_r:writeString(tp_all_distance[i]..'\n')
        end
        fp_r:close()
    end

    -- plot the graph
    local plot_length = 4*total_manualpick
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
    local graph_name=paths.concat(opt.debugdir,"autopick.png")
    gnuplot.pngfigure(graph_name)
    gnuplot.plot({'probability',x_value,probability,'-'},{'recall',x_value,recall,'-'},{'precision',x_value,precision,'-'},{'distance',x_value,averageDistance,'-'})		
    gnuplot.xlabel('distance is normalized by 50')
    gnuplot.grid(true)
    gnuplot.title('Recall-Precision-Probability-Distance')
    gnuplot.plotflush()
end

if opt.save_coordinate then
    -- load results file
    local autopick_file_coordinate=torch.load(inputFiles)
    local dirname = paths.dirname(inputFiles)
    local autopick_file_final = {}
    for i=1,#autopick_file_coordinate do
        local mrc_filename=autopick_file_coordinate[i][1][4]
        local file_coordinate = {}
        for j=1, #autopick_file_coordinate[i] do
            --if autopick_file_coordinate[i][j][3] >= opt.save_threshold then
            if autopick_file_coordinate[i][j][3] >= 0.1 and autopick_file_coordinate[i][j][3] <= 0.5 then
                table.insert(file_coordinate, autopick_file_coordinate[i][j])
            end
        end
        table.insert(autopick_file_final, file_coordinate)
    end
    write_coordinate(autopick_file_final, dirname, opt.particle_size, opt.save_coordinateType, opt.save_coordinateSymbol) 
end

if opt.save_prediction_score then
    local autopick_file_coordinate=torch.load(inputFiles)
    local coordinate_prediction_score = {}
    for i=1,#autopick_file_coordinate do
        for j=1, #autopick_file_coordinate[i] do
            table.insert(coordinate_prediction_score, autopick_file_coordinate[i][j][3])
        end
    end
    local total_results_file = paths.concat(opt.debugdir,'results_prediction_score.txt')
    local fp_r = torch.DiskFile(total_results_file,"w")
    for i=1,#coordinate_prediction_score do
        fp_r:writeString(coordinate_prediction_score[i]..'\n')
    end
    fp_r:close()
end

