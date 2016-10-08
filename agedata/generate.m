% fid=fopen('example.mat');
%
% data=fread(fid,256*256,'float64');
% fclose(fid)

data=h5read('sample.h5','/Argento_13_new4/Argento_13_new4 data');
valmin=-0.4;
valmax=0.4;

%data=rand(256,256)*0.01;
sd1=size(data,1);
sd2=size(data,2);
data(data<valmin | data>valmax)=NaN;
base=reshape(data,[sd1 sd2]);

% for k=1:300
%     x1=randi(sd1);
%     y1=randi(sd2);
%     x2=max(1,min(sd1,x1+randi(200)-100));
%     y2=max(1,min(sd2,y1+randi(200)-100));
%
%     rpts = linspace(x1,x2,1000);
%     cpts = linspace(y1,y2,1000);   %# A set of column points for the line
%     index = sub2ind(size(base),round(rpts),round(cpts));  %# Compute a linear index
%     base(index) = base(index)+0.02;
%
% end
%base(base<valmin | base>valmax)=NaN;
% imshow(base,[-0.4 0.4])
% impixelinfo
% colormap hsv
% waitforbuttonpress
h5create('agingsamples.h5','/Argento13_0/Argento13_0 data',[sd1 sd2]);
h5create('agingsamples.h5','/Argento13_0/Argento13_0 MASK',[sd1 sd2],'DataType','int8','FillValue',int8(0));
h5write('agingsamples.h5','/Argento13_0/Argento13_0 data',data);

pat=zeros(size(base));
for j=1:110
    h=fspecial('gaussian',11);
    se=strel('square',3);
    
    for t=1:100
        %t
        
        for i=1:1000
            seed(i,1)=randi(sd1);
            seed(i,2)=randi(sd2);
        end
        
        
        if(rand >0.2)
            x1=randi(sd1);
            y1=randi(sd2);
            x2=max(1,min(sd1,x1+randi(200)-100));
            y2=max(1,min(sd2,y1+randi(200)-100));
            
            rpts = linspace(x1,x2,1000);
            cpts = linspace(y1,y2,1000);   %# A set of column points for the line
            index = sub2ind(size(base),round(rpts),round(cpts));  %# Compute a linear index
            base(index) = base(index)+0.02;
        end
        
        
        for i=1:1000
            pat(seed(i,1), seed(i,2)) = pat(seed(i,1), seed(i,2)) + 0.01;
        end
        out=base-pat;
    end
    out=imfilter(out,h);
    out=min(out,base);
    mi=min(min(out));
    mi=mi-0.02;
    out=max(out,mi);
    size(out(out<valmin | out>valmax),1)
    out(out<valmin | out>valmax)=NaN;
    %     imshow(out,[-0.4 0.4])
    %     impixelinfo
    %     colormap hsv
    %     waitforbuttonpress
    s=['/Argento13_' int2str(j)];
    h5create('agingsamples.h5',[s s ' data'],[sd1 sd2]);
    h5create('agingsamples.h5',[s s ' MASK'],[sd1 sd2],'DataType','int8','FillValue',int8(0));
    h5write('agingsamples.h5',[s s ' data'],out);
    j
end

%
%
% imshow(out,[-0.4 0.4])
% colormap hsv