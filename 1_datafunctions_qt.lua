require 'qtwidget'
require 'qttorch'
require 'image'

function display(mrc,coordinate,particle_size,filename,color)
        --image.display(mrc)
        if not color then
                color = "white"
        end
        local qimage = qt.QImage.fromTensor(mrc[1])
        local painter_image = qtwidget.newimage(qimage)
	if color == "test" then
        	for i=1,#coordinate do
                        if coordinate[i][3]>0.95 then
                                if coordinate[i][5] == 1 then
                                        painter_image:arc(coordinate[i][1],coordinate[i][2],particle_size/2,0,360)
                                        painter_image:setcolor("red")
                                        painter_image:setlinewidth(2)
                                        painter_image:stroke()
                                else
                                        painter_image:arc(coordinate[i][1],coordinate[i][2],particle_size/2,0,360)
                                        painter_image:setcolor("blue")
                                        painter_image:setlinewidth(2)
                                        painter_image:stroke()
                                end
                        end
                end
	else
		for i=1,#coordinate do
                	painter_image:arc(coordinate[i][1],coordinate[i][2],particle_size/2,0,360)
                	painter_image:setcolor(color)
                	painter_image:setlinewidth(2)
                	painter_image:stroke()
        	end
	end
        painter_image:write(filename)
        painter_image:close()
        --return painter_image:image():toTensor():clone()       
end

function display_compare(mrc,coordinate1,coordinate2,particle_size,filename,symbol,color1,color2)
	-- coordinate 1 refer manual pick, color red
	-- coordinate 2 refer autopick , color blue, if is true positive, color black
	-- symbol==0 , plot all the particles
	-- symbol==1 , plot only the matched particles both
	-- symbol==2 , plot the all the coordinate1 as red color, plot the matched coordinate2 as black, plot the unmatched coordinate2 as blue 
        if not color1 then
                color1 = "red"
        end
        if not color2 then
                color2 = "blue"
        end
	if not symbol then
		symbol = 0
	end

        local qimage = qt.QImage.fromTensor(mrc[1])
        local painter_image = qtwidget.newimage(qimage)
        for i=1,#coordinate1 do
		if symbol==1 then
			if coordinate1[i][5] == 1 then
                		painter_image:arc(coordinate1[i][1],coordinate1[i][2],particle_size/2,0,360)
                		painter_image:setcolor(color1)
                		painter_image:setlinewidth(2)
                		painter_image:stroke()
			end
		else
                	painter_image:arc(coordinate1[i][1],coordinate1[i][2],particle_size/2,0,360)
                	painter_image:setcolor(color1)
                	painter_image:setlinewidth(2)
                	painter_image:stroke()
		end
        end
        for i=1,#coordinate2 do
		if symbol==1 then 
			if coordinate2[i][5] == 1 then
                		painter_image:arc(coordinate2[i][1],coordinate2[i][2],particle_size/2,0,360)
                		painter_image:setcolor(color2)
                		painter_image:setlinewidth(2)
                		painter_image:stroke()
			end
		elseif symbol==2 then
			if coordinate2[i][3]>=0.95 then
				if coordinate2[i][5] == 1 then
					color2 = "black"	
				else
					color2 = "blue"
				end
                		painter_image:arc(coordinate2[i][1],coordinate2[i][2],particle_size/2,0,360)
                		painter_image:setcolor(color2)
                		painter_image:setlinewidth(2)
                		painter_image:stroke()
			end
		else
			if coordinate2[i][5] == 1 then
				color2 = "black"	
			else
				color2 = "blue"
			end
                	painter_image:arc(coordinate2[i][1],coordinate2[i][2],particle_size/2,0,360)
                	painter_image:setcolor(color2)
                	painter_image:setlinewidth(2)
                	painter_image:stroke()
		end
        end

        painter_image:write(filename)
        painter_image:close()
end

function display_cross(mrc,coordinate,filename,length,color)
        if not color then
                color="black"
        end
        if not length then
                length = 10
        end
        local qimage = qt.QImage.fromTensor(mrc[1]:float():clone())
        local painter_image = qtwidget.newimage(qimage)
        for i=1,#coordinate do
		local coor_x = coordinate[i][1]
		local coor_y = coordinate[i][2]
                local x1 = coor_x-math.ceil(length/2)
                if x1<1 then x1 = 1 end
                local x2 = coor_x+math.ceil(length/2)
                if x2>mrc:size(3) then x2 = mrc:size(3) end
                painter_image:moveto(x1,coor_y)
                painter_image:lineto(x2,coor_y)

                local y1 = coor_y-math.ceil(length/2)
                if y1<1 then y1 = 1 end
                local y2 = coor_y+math.ceil(length/2)
                if y2>mrc:size(2) then y2 = mrc:size(2) end
                painter_image:moveto(coor_x,y1)
                painter_image:lineto(coor_x,y2)
                painter_image:setcolor(color)
                painter_image:setlinewidth(1)
                painter_image:stroke()
        end
        print(filename)
        --print(string.sub(filename,1,string.len(filename)-1))
        painter_image:write(filename)
        painter_image:close()
end

function display_cross_compare(mrc,coordinate_manual,coordinate_auto,filename,length,color1,color2)
        if not color1 then
                color1="red"
        end
        if not color2 then
                color2="blue"
        end
        if not length then
                length = 10
        end
        local qimage = qt.QImage.fromTensor(mrc[1]:float():clone())
        local painter_image = qtwidget.newimage(qimage)
        for i=1,#coordinate_manual do
		local coor_x = coordinate_manual[i][1]
		local coor_y = coordinate_manual[i][2]
                local x1 = coor_x-math.ceil(length/2)
                if x1<1 then x1 = 1 end
                local x2 = coor_x+math.ceil(length/2)
                if x2>mrc:size(3) then x2 = mrc:size(3) end
                painter_image:moveto(x1,coor_y)
                painter_image:lineto(x2,coor_y)

                local y1 = coor_y-math.ceil(length/2)
                if y1<1 then y1 = 1 end
                local y2 = coor_y+math.ceil(length/2)
                if y2>mrc:size(2) then y2 = mrc:size(2) end
                painter_image:moveto(coor_x,y1)
                painter_image:lineto(coor_x,y2)
                painter_image:setcolor(color1)
                painter_image:setlinewidth(1)
                painter_image:stroke()
        end
        for i=1,#coordinate_auto do
		local coor_x = coordinate_auto[i][1]
		local coor_y = coordinate_auto[i][2]
                local x1 = coor_x-math.ceil(length/2)
                if x1<1 then x1 = 1 end
                local x2 = coor_x+math.ceil(length/2)
                if x2>mrc:size(3) then x2 = mrc:size(3) end
                painter_image:moveto(x1,coor_y)
                painter_image:lineto(x2,coor_y)

                local y1 = coor_y-math.ceil(length/2)
                if y1<1 then y1 = 1 end
                local y2 = coor_y+math.ceil(length/2)
                if y2>mrc:size(2) then y2 = mrc:size(2) end
                painter_image:moveto(coor_x,y1)
                painter_image:lineto(coor_y,y2)
                painter_image:setcolor(color2)
                painter_image:setlinewidth(1)
                painter_image:stroke()
        end
        painter_image:write(filename)
        painter_image:close()
end

