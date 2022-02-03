function Asub = MinSubset(A, tol)

D = squareform(pdist(A'));


while sum(D < tol,'all')>length(D) 
    
    flag = 0;
    for i = 1:length(D)
        
        same_vecs = setdiff(find(D(i,:)<tol),i);
        
        if ~isempty(same_vecs) && flag == 0
            same_vecs_final = same_vecs;
            flag = 1;
        end
        
    end
    
    A(:,same_vecs_final) = [];
    D(:,same_vecs_final) = [];
    D(same_vecs_final,:) = [];
    
end    

Asub = A;
