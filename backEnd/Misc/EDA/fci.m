function Img = fci(HSI)

hcube = hypercube(HSI,1:size(HSI,3)); 
Img = colorize(hcube);
figure;
imagesc(Img)
axis equal off

end
