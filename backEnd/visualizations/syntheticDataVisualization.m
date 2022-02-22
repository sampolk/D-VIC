%% Synthetic HSI Visualization

%% 
load('syntheticHSI.mat')

%% Figure 6

figure

colors = {'k', '#4658F8', '#2896EB', '#13BEB8', '#80CA57', '#FCBB3D'  };
rgb_key = [[0,0,0]; [0.275,0.345,0.973]; [15.7,58.8, 92.2]./100; [7.5, 74.5, 72.2]./100; [50.2,79.2,34.1]./100; [98.8, 73.3, 23.9]./100 ];

c_data = zeros(M,N, 3);
for i = 1:M
    for j = 1:N
        c_data(i,j,:) = rgb_key(GT(i,j)+1,:);
    end
end

subplot(1,2,1)
image(c_data)
title('Synthetic Data Ground Truth', 'interpreter', 'latex')
xticks([])
yticks([])
pbaspect([1,1,1])
set(gca,'FontName', 'Times', 'FontSize', 20)

subplot(1,2,2)
hold on 
for k = 1:length(unique(Y))
    
    XkIdx = find(Y == k-min(Y)-1);

    for i = 1:5

        sample = randsample(length(Xk),1);
        c = colors(k);
        plot(1:D, X(XkIdx(sample),:),'Color', c{1})

    end
end
axis tight
box on
title('Randomly Selected Pixel Spectra, Colored by Class', 'interpreter', 'latex')
xlabel('Spectral Band Number')
ylabel('Reflectance')

pbaspect([1,1,1])
set(gca,'FontName', 'Times', 'FontSize', 20)