----------------------------------------------------------
-- Author WangFeng  2016/04/14
-- This script can take the script "e2proc2d.py" in EMAN2.12 to do preprocess to micrograph.

-- Input:
--    -inputDIR: the dir of mrc files
--    -outputDIR: the output of the filtered mrc file
--    -number: number of mrc files to preprocess, if number <=0 then filter all micrograph
--    -sigma: gaussian lowpass sigma

--------------------------------------------------------
require 'torch'

if not opt then 
    print '==> mrc2jpg options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('mrc file or dir to jpg')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-inputDIR', 'none', 'the input dir of mrc files')
    cmd:option('-outputDIR', 'none', 'the output dir of mrc files')
    cmd:option('-number', -1, 'number of images to show,if equal to 0 then show all images')
    cmd:option('-sigma', 0.05, 'sigma of the gaussian filter')
    opt=cmd:parse(arg or {})
end

local mrc_files = {}
if paths.dirp(opt.inputDIR) then
    for file in paths.files(opt.inputDIR) do
        if string.match(file,"%w+.mrc") then
            local mrc_filename = paths.concat(opt.inputDIR, file)
            table.insert(mrc_files,mrc_filename)
        end	
    end
else
    error("Wrong input, there is not mrc files in the dir:",opt.inputDIR)
end
table.sort(mrc_files)

if not paths.dirp(opt.outputDIR) then
    os.execute('mkdir '..opt.outputDIR)
end
if opt.number <=0 then
    mrc_number = #mrc_files
else
    mrc_number = opt.number
end

for i=1, mrc_number do
    local filename = paths.basename(mrc_files[i])
    local output_filename = paths.concat(opt.outputDIR, filename)    
    os.execute('e2proc2d.py '..mrc_files[i]..' '..output_filename..' --process=filter.lowpass.tanh:sigma='..opt.sigma)
end
