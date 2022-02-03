%% 

files = {'DVISResultsSalinasACorrectedATGP',...
    'DVISResultsSalinasACorrectedAVMAX', ...
    'DVISResultsSalinasACorrectedMVC_NMF', ... 
    'DVISResultsSalinasACorrectedN_FINDR', ...
    'DVISResultsSalinasACorrectedPLM', ...
    'DVISResultsSalinasACorrectedPPI', ... 
    'DVISResultsSalinasACorrectedSGA', ... 
    'DVISResultsSalinasACorrectedVCA'};

algs = cellfun( @(x) erase(x, 'DVISResultsSalinasACorrected'), files, 'UniformOutput', false);
%% 
OAOpt = zeros(length(algs),1);
OAStd = zeros(length(algs),1);
Ctemp = zeros(83*86, length(algs));

load('SalinasACorrected')
%%
for i = 2

    load(files{i})

    M = mean(OAs, 3); 

    [OAOpt(i), k]  = max(M(:));
    [j,l] = ind2sub(size(M), k);
    OAStd(i) = std(squeeze(OAs(j,l,:)));

    [~,k] = min(abs(squeeze(OAs(j,l,:))-OAOpt(i)));

    Ctemp(:,i) = Cs(:,j,l,k);
end

%%
% h = figure;
for i = 1:7

    subplot(2,4,i)
    eda(Ctemp(:,i), 0, Y)
    title(['D-VIS, Using ' algs{i} ' Unmixing'], 'interpreter', 'latex', 'FontSize', 16)
end
saveas(h, 'OptimalClusterings', 'jpeg')

