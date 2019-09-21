datadir = uigetdir;
out_file= strcat(datadir, '/' ,'manual_seg_test.csv');

fnames=dir(strcat(datadir, '/*.png'));
Nfnames=length(fnames);

set(0,'units','pixels')  
%Obtains this pixel information
Pix_SS = get(0,'screensize');
screenWidth = Pix_SS(3);
screenHeight= Pix_SS(4);
screen_scale = 1.5;

do_zoom = 0  % to zoom or not to zoom image - DON'T zoom with new focused hFig!
storedvalues=cell(Nfnames, 19);  % 1 filename + 3*6 coords
out=0; out2=0; out3=0; out4=2; % initial
for i=1:Nfnames
    filename = fnames(i).name;
    framename = filename;
    A=imread(strcat(fnames(i).folder, '/', fnames(i).name)); 

    while (out4>0)
        x1=0;y1=0; x2=0;y2=0; x3=0;y3=0;    % reset to avoid peeking
        x4=0;y4=0; x5=0;y5=0; x6=0;y6=0;
        x7=0;y7=0; x8=0;y8=0; x9=0;y9=0;
        
        % redraw image each redo   
%         figure(1); clf();
        hFig = figure('NAME','IOP Segmenter',...
            'Position', [0 0 screenWidth screenHeight]); 
        fpos = get(hFig,'Position');
        if ~do_zoom
    %         imshow(A, 'InitialMagnification', 500); hold on; 
%             imshow(A, 'InitialMagnification','fit'); hold on; 
%             axOffset = (fpos(3:4)-[size(A,2) size(A,1)])/2;
            axOffset = (fpos(3:4)-[size(A,2)/screen_scale size(A,1)/screen_scale])/2;
            ha = axes('Parent',hFig,'Units','pixels',...
                        'Position',[axOffset size(A,2)/screen_scale size(A,1)/screen_scale]);
            hImshow = imshow(A,'Parent',ha); hold on;
%         else   % zoom into correct area
%             A_blue = A(:,:,3);  % matlab imread defaults to RGB
% %             rowMeans = mean(A_blue, 2);
% %             colMeans = mean(A_blue, 1);
%             rowMax = max(A_blue, [], 2);
%             colMax= max(A_blue, [], 1);
%             rowPercentiles = prctile(rowMax, [5, 10, 25, 50, 75, 90, 95]);
%             rowCond = rowMax > rowPercentiles(4);
%             rowStart = find(rowCond==1, 1);
%             rowEnd = find(rowCond==1, 1, 'last');
% 
%             colPercentiles = prctile(colMax, [5, 10, 25, 50, 75, 90, 95]);
%             colCond = colMax > colPercentiles(3);
%             colStart = find(colCond==1, 1);
%             colEnd = find(colCond==1, 1, 'last');
% 
%             B = A(rowStart:rowEnd, colStart:colEnd, :);
%             size(B)
%             axOffset = (fpos(3:4)-[size(B,2)/screen_scale size(B,1)/screen_scale])/2;
%             ha = axes('Parent',hFig,'Units','pixels',...
%                         'Position',[axOffset size(B,2)/screen_scale size(B,1)/screen_scale]);
%             hImshow = imshow(B,'Parent',ha); hold on;
        end % if
        title(framename)    % for easier reference

        out=0;
        %uiwait(msgbox('if correctly identified outer ring, press a key, otherwise click'));
        %if waitforbuttonpress==0 %0 if it detects a click, 1 keypress
        uiwait(msgbox('1. Locate the outer ring by clicking 3 locations along the arc\n; 2. Locate the left and right inner ring similarly'));
        if out==0
%             uiwait(msgbox('Locate the outer ring by clicking 3 locations along the ring'));
            [x1,y1] = ginput(1); 
            plot(x1,y1,'r*', 'MarkerSize', 5);
            [x2,y2] = ginput(1);
            plot(x2,y2,'r*', 'MarkerSize', 5);
            [x3,y3] = ginput(1);
            plot(x3,y3,'r*', 'MarkerSize', 5);
        end

%         answer2 = questdlg('Is the left inner ring...', ...
%         'Left inner ring', ...
%         'Not present','Present','Present');
%         % Handle response
%         switch answer2
%             case 'Not present'
%                 out2 = 0;
%             case 'Present'
%                 out2 = 2;
%         end    
%         if out2==0
%             x4=0;y4=0; x5=0;y5=0; x6=0;y6=0;
%             x7=0;y7=0; x8=0;y8=0; x9=0;y9=0;
%         end
        out2 = 2;
        if out2==2  % left inner ring
%             uiwait(msgbox('Locate the left inner ring by clicking 3 locations along the ring'));
            [x4,y4] = ginput(1);
            plot(x4,y4,'m*', 'MarkerSize', 5);
            [x5,y5] = ginput(1);
            plot(x5,y5,'m*', 'MarkerSize', 5);
            [x6,y6] = ginput(1);
            plot(x6,y6,'m*', 'MarkerSize', 5);
        end

%         answer3 = questdlg('Is the right inner ring...', ...
%         'Right inner ring', ...
%         'Not present','Present','Present');
%         % Handle response
%         switch answer3
%             case 'Not present'
%                 out3 = 0;
%             case 'Present'
%                 out3 = 2;
%         end  
%         if out3==0
%             x7=0;y7=0;x8=0;y8=0;x9=0;y9=0;
%         end
        out3=2;
        if out3==2  % right inner ring
%             uiwait(msgbox('Locate the right inner ring by clicking 3 locations along the ring'));
            [x7,y7] = ginput(1);
            plot(x7,y7,'w*', 'MarkerSize', 5);
            [x8,y8] = ginput(1);
            plot(x8,y8,'w*', 'MarkerSize', 5);
            [x9,y9] = ginput(1);
            plot(x9,y9,'w*', 'MarkerSize', 5);
        end
        
        %% overplot circles
        %[cx1, cy1, r1] = circle_from_3pts([x1, y1], [x2, y2], [x3, y3]);
        %[cx2, cy2, r2] = circle_from_3pts([x4, y4], [x5, y5], [x6, y6]);
        %[cx3, cy3, r3] = circle_from_3pts([x7, y7], [x8, y8], [x9, y9]);
        %draw_circle(cx1, cy1, r1, 'red');
        %draw_circle(cx2, cy2, r2, 'magenta')
        %draw_circle(cx3, cy3, r3, 'white')
        
        % confirm answers
        answer4 = questdlg('Are you happy with your segmentation...', ...
        'Segmentation good', ...
        'Happy','Need to redo','Need to redo');
        % Handle response
        switch answer4
            case 'Happy'
                out4 = 0;
            case 'Need to redo'
                out4 = 2;
        end
    end %while
    
    out4 = 2;   % reset for new image
    hold off;
    close;
    storedvalues{i,1}=framename; 
    if do_zoom
        storedvalues{i,2}=x1 + rowStart; storedvalues{i,3}=y1+colStart;
        storedvalues{i,4}=x2 + rowStart; storedvalues{i,5}=y2+colStart;
        storedvalues{i,6}=x3 + rowStart; storedvalues{i,7}=y3+colStart;
        if out2
            storedvalues{i,8}=x4 + rowStart; storedvalues{i,9}=y4+colStart;
            storedvalues{i,10}=x5 + rowStart; storedvalues{i,11}=y5+colStart;
            storedvalues{i,12}=x6 + rowStart; storedvalues{i,13}=y6+colStart;
        else % just initial values
            storedvalues{i,8}=x4; storedvalues{i,9}=y4;
            storedvalues{i,10}=x5; storedvalues{i,11}=y5;
            storedvalues{i,12}=x6; storedvalues{i,13}=y6;
        end
        if out3
            storedvalues{i,14}=x7 + rowStart; storedvalues{i,15}=y7+colStart;
            storedvalues{i,16}=x8 + rowStart; storedvalues{i,17}=y8+colStart;
            storedvalues{i,18}=x9 + rowStart; storedvalues{i,19}=y9+colStart;
        else % just initial values
            storedvalues{i,14}=x7; storedvalues{i,15}=y7;
            storedvalues{i,16}=x8; storedvalues{i,17}=y8;
            storedvalues{i,18}=x9; storedvalues{i,19}=y9;
        end
    else        
        storedvalues{i,2}=x1; storedvalues{i,3}=y1;
        storedvalues{i,4}=x2; storedvalues{i,5}=y2;
        storedvalues{i,6}=x3; storedvalues{i,7}=y3;
        storedvalues{i,8}=x4; storedvalues{i,9}=y4;
        storedvalues{i,10}=x5; storedvalues{i,11}=y5;
        storedvalues{i,12}=x6; storedvalues{i,13}=y6;
        storedvalues{i,14}=x7; storedvalues{i,15}=y7;
        storedvalues{i,16}=x8; storedvalues{i,17}=y8;
        storedvalues{i,18}=x9; storedvalues{i,19}=y9;
    end % if for adjusting zoom coords
    
    % write every frame to avoid losing data
    % Convert cell to a table and use first row as variable names
    T = cell2table(storedvalues(1:i,:));
    writetable(T, out_file, 'WriteVariableNames',0) % Write the table to a CSV file
end % for

% % Convert cell to a table and use first row as variable names
% T = cell2table(storedvalues);
% writetable(T, out_file, 'WriteVariableNames',0) % Write the table to a CSV file

function [cx, cy, radius] = circle_from_3pts(b, c, d)
    temp = c(1)^2 + c(2)^2;
    bc = (b(1)^2 + b(2)^2 - temp) / 2;
    cd = (temp - d(1)^2 - d(2)^2) / 2;
    det = (b(1) - c(1)) * (c(2) - d(2)) - (c(1) - d(1)) * (b(2) - c(2));

    if abs(det) < 1.0e-10
        cx = 0;
        cy = 0;
        radius = 0;
    else
        % Center of circle
        cx = (bc*(c(2) - d(2)) - cd*(b(2) - c(2))) / det;
        cy = ((b(1) - c(1)) * cd - (c(1) - d(1)) * bc) / det;
        radius = ((cx - b(1))^2 + (cy - b(2))^2)^(0.5);
    end
end

function h = draw_circle(x,y,r, color)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit, color);
end