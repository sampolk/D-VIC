function eda(x, log_on, Y)

if length(x) == 7138 % Salinas A
    M = 83;
    N = 86;
elseif length(x) == 250000 % AVIRIS-NG Subset
    M = 500;
    N = 500;
elseif length(x) == 1701540 % Full AVIRIS-NG Image
    M = 3151;
    N = 540;
elseif length(x) == 314368 % Kennedy Space Center
    M = 512;
    N = 614;
elseif length(x) == 377856 % Botswana
    M = 1476;
    N = 256;
elseif length(x) == 21025 % Indian Pines
    M = 145;
    N = 145;
elseif length(x) == 783640 % Pavia Centre
    M = 1096;
    N = 715;
elseif length(x) == 207400 % Pavia University
    M = 610;
    N = 340;
elseif length(x) == 111104 % Salinas
    M = 512;
    N = 217;
elseif length(x) == 2803351
    M = 1601;
    N = 1751;
elseif length(x) == 206976
    M = 448;
    N = 462;
elseif length(x) == 1e4
    M = 100;
    N = 100;
else
    M = NaN;
end
if length(x) == 1e4
    flag = 0;
else
    flag = 1;
end

if ~isnan(M)
    if nargin == 1
        log_on = 0;
    end
    if nargin == 3
        if flag == 1
            z = zeros(size(x));
            x = alignClusterings(Y(Y>1)-1,x(Y>1));
            z(Y>1) = x+1;
            x = z;
        else
             x = alignClusterings(Y,x);
        end
    end
    if log_on
        try
            imagesc(log10(reshape(x,M,N)))
        catch
            imagesc((reshape(x,M,N)))
        end
    else
        imagesc((reshape(x,M,N)))
    end
    xticks([])
    yticks([])
    axis equal tight
else
    disp('Error: eda.m function not compatible with inputted dataset.')
end