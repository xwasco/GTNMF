function [accuracy,CE,CC,ass]=classificationAccuracy(gt,clustering)
M1=zeros(size(gt,1),max(gt(:)));
M1(sub2ind(size(M1),1:size(gt,1),gt'))=1;
%clustering=clustering+1;
M2=zeros(size(clustering,1),max(clustering(:)));
M2(sub2ind(size(M2),1:size(clustering,1),clustering'))=1;
CE=M1'*M2;
[ass,cost]=munkres(-CE);
%cost=cost+(gt==1)'*(clustering==2 | clustering;
accuracy=-cost/numel(gt);

if nargout>=3
    CC=zeros(size(CE));
    for i=1:size(CC,1)
        if ass(i)>0
            CC(:,i)=CE(:,ass(i));    
        end
    end
end

end