function plot_gamma0_decay()
%PLOT_GAMMA0_DECAY Compare the two predicted gamma=0 decay rates.

root = fileparts(mfilename('fullpath'));
A = readmatrix(fullfile(root,'long_4x3_e0.dat'), ...
               'FileType','text','CommentStyle','#');
B = readmatrix(fullfile(root,'long_4x3_e01.dat'), ...
               'FileType','text','CommentStyle','#');

blue = [0 0.4470 0.7410];
red = [0.8500 0.3250 0.0980];
lightBlue = [0.62 0.80 0.91];
lightRed = [0.95 0.72 0.58];

figure('Color','w');
tref = logspace(3,7,100);
c1 = interp1(A(:,1),A(:,2),1e4)*1e4;
c2 = interp1(B(:,1),B(:,2),1e4)*sqrt(1e4);

% Draw wide reference lines first. The narrower broken numerical lines
% remain visible on top, while their gaps expose the references below.
hSlope1 = loglog(tref,c1./tref,'-','Color',lightBlue, ...
                 'LineWidth',2.4); hold on;
hSlope2 = loglog(tref,c2./sqrt(tref),'-','Color',lightRed, ...
                 'LineWidth',2.4);
hData1 = loglog(A(:,1),A(:,2),'--','Color',blue,'LineWidth',1.4);
hData2 = loglog(B(:,1),B(:,2),'-.','Color',red,'LineWidth',1.4);

xlabel('t');
ylabel('E(t)');
legend([hData1 hData2 hSlope1 hSlope2], ...
       '\epsilon_{stab}=0','\epsilon_{stab}=0.1', ...
       'slope -1','slope -1/2','Location','southwest');
grid on;
axis tight;
exportgraphics(gcf,fullfile(root,'gamma0_decay_reproduced.pdf'), ...
               'ContentType','vector');
end
