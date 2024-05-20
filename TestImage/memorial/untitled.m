t = [32 8 2 1 0.5];
l1 = [];
l2 = [];
l3 = [];
for i = [1 3 5 6 7]
A = double(imread("memorial006"+i+".png"));
imagesc(A(:,:,1))
l1 = [l1 A(100,150,1)];
l2 = [l2 A(500,400,1)];
l3 = [l3 A(300,20,1)];
end
%%
plot(log(l1/255)+log(t),l1,'o-');hold on
plot(log(l2/255)+log(t),l2,'o-');
plot(log(l3/255)+log(t),l3,'o-');hold off