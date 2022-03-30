
%GENERAL LINEAR OPTIMIZATION STANDARD FORM PROBLEM :
% min of f:(xj)->sum(wj.*xj)
% s.t. (n conditions) :
% sum(a1j.*xj)= b1
% . . .= . . .
% sum(anj.*xj)= bn
% xi ≥ 0 for all i
%meaning A=(aij)i,j ; b=(bi)i ; c=(wj)j
%and also min c'*xi s.t. Ax=b

%RESOLVE THE FOLLOWING PROBLEM :
% min of f:(x1,x2)->2*x1 + 3*x2
% s.t. :
% 4x1 + 2x2 − x3 = 12
% x1 + 4x2 − x4 = 6
% x1 ≥ 0, . . . , x4 ≥ 0

A=[4 2 -1 0;1 4 0 -1];
b=[12;6];
c=[2;3;0;0];

%Two phase if a basic feasible solution is not easy to find
[costOpt, xOpt, BOpt] = LP_Two_Phase_Simplex(A,b,c)


function [costOpt, xOpt, BOpt] = LP_Two_Phase_Simplex(A,b,c)
    % Phase 1 : resolving a virtual problem to get the 
    %initial basic feasible solution
    [m,n]=size(A);
    B_virtual=n+1:n+m; % Identical matrix base
    init_A = [A eye(m)];% Concatenate identical matrix
    init_c=[zeros(n,1);ones(m,1)]; % Define virtual objective function
    [cano_tab, ~, ~, B_init] = LP_Simplex(init_A,b,init_c,B_virtual);

    % Phase 2 : Return to the unvirtual function
    cano_tab(:,B_virtual)=[]; %Delete virtual columns
    tableau=[cano_tab(1:end-1,:) ; c' 0]; % Create tableau
    cano_tab_phase2=[eye(m) zeros(m,1) ; -tableau(end,B_init) 1]*[inv(tableau(1:end-1,B_init)) zeros(m,1); zeros(1,m) 1]*tableau;
    %Created the canonical tableau
    xOpt=zeros(1,n);
    %Push solutions with the right base
    xOpt(B_init)=cano_tab_phase2(1:end-1, end);
    costOpt=xOpt*c;
    BOpt=B_init;
end


function [cano_tab, costOpt, xOpt, BOpt] = LP_Simplex(A,b,c,v)
    % One phased method
    [m,n]=size(A);
    tableau = [A b ; c' 0];
    cano_tab=[eye(m) zeros(m,1) ; -tableau(end,v) 1]*[inv(tableau(1:end-1,v)) zeros(m,1); zeros(1,m) 1]*tableau;
    BOpt=[];
    while sum(cano_tab(end,:)>=0)~=size(cano_tab,2) %While solution is not BFS
        [~, q]=min(cano_tab(end,1:n)); %Compute min of last row
        BOpt(end+1)=q; %Stack the new base
        if sum(cano_tab(1:m,q)>0) < 1 %Check for full negative values
            disp("No solution : the problem is unbounded")
            break
        else
            yiq=cano_tab(1:m,q);
            yiq(yiq<0) = 0; %Remove negative elements
            [~, p]=min(cano_tab(1:m,end)./yiq); %Compute the min of column divided by column pivot
            pivot=cano_tab(p,q); %Get pivot value
            rowsInd=(1:m+1); rowsInd(p)=[]; %Prepare for rows operations
            cano_tab(p,:)=cano_tab(p,:)/pivot; %Normalize pivot row
            for row=rowsInd %For each row that is not the pivot's row 
                cano_tab(row,:) = cano_tab(row,:) - cano_tab(row,q)*cano_tab(p,:);
            end 
        end
    end
    xOpt=zeros(1,n-1);
    xOpt(BOpt)=cano_tab(1:end-1, end);
    costOpt=cano_tab(end,end);
end


%Brute forcing the simplest BFS
function [costOpt, xOpt, BOpt] = LP_Bourrin(A,b,c)
    [m,n] = size(A);
    All_ind= nchoosek(1:n,m);
    solutions=zeros(n,size(All_ind,1));
    for i=1:size(All_ind,1)
        B= A(:,All_ind(i,:));
        x=zeros(1, n)';
        x(All_ind(i,:))=(B\b)';
        if sum(x>=0)== n
            solutions(:,i)= x; %Stack solutions into a matrix
        end
    end
    [costOpt, idxmin] = min(c'*solutions); %Compute min over all solutions
    xOpt=solutions(:,idxmin);
    BOpt=All_ind(idxmin,:);
end 
