require 'torch'
require '1_datafunctions'

if not opt then
    print '==> analysis the shift of particles in relion 2D results.'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Options:")
    cmd:option('-input', 'none', 'a input DIR of starfile')
    cmd:option('-class_shift', false, 'whether to compute the average shift based on the class')
    opt = cmd:parse(arg or {})
end

local input = opt.input
local star_files = {}
if paths.dirp(input) then
    for file in paths.files(input) do
        if string.match(file, "%.star$") then
            local star_filename = paths.concat(input, file)
            table.insert(star_files, star_filename)
        end    
    end
elseif paths.filep(input) and string.match(input, "%.star$") then
    table.insert(star_files, input)
else
    error("Wrong input, there is not mrc files in the dir! Or it not a mrc file!")
end
table.sort(star_files)
print(star_files)
target_label = {"rlnClassNumber", "rlnOriginX", "rlnOriginY"}
for i=1, #star_files do
    print(star_files[i])
    local value_all_sample = read_star(star_files[i], target_label) 
    local all_shift = 0
    local all_number = 0
    local class_average_shift = {}
    local class_number = {}
    for j=1, #value_all_sample do
        local class_index = tonumber(value_all_sample[j][1])
        local coordinate_x = tonumber(value_all_sample[j][2])
        local coordinate_y = tonumber(value_all_sample[j][3])
        local shift = math.sqrt(coordinate_x^2 + coordinate_y^2)
        all_shift = all_shift + shift
        all_number = all_number + 1
        if class_average_shift[class_index] then 
            class_average_shift[class_index] = class_average_shift[class_index] + shift
            class_number[class_index] = class_number[class_index] + 1
        else
            class_average_shift[class_index] = shift
            class_number[class_index] = 1
        end
    end
    all_shift = all_shift/all_number
    print("all average shift:", all_shift)
    print("class average shift:")
    for j=1, #class_average_shift do
        if class_number[j] then
            print("class:"..j.."\tshift:"..(class_average_shift[j]/class_number[j]))
        end
    end
end
