%%
% utility script to plot scenario settings for single instances
% Note: assumes TruckSim .mat file is preloaded.
% TODO: Revise for manual .mat input
close all
clc

figure()
plot(Xo, Yo, LineWidth=1.5, HandleVisibility='off')
hold on
plot(Xo(1),Yo(1), '*', color='g', DisplayName='start')
plot(Xo(end),Yo(end), '*', color='r', DisplayName='end')
hold off
legend
axis("equal")
xlabel('X (m)')
ylabel('Y (m)')
grid
set(gcf,'color','w')

figure
plot(T_Event, Vx/1.609, LineWidth=1.5)
xlabel('Time (s)')
ylabel('Velocity (mph)')
grid
set(gcf,'color','w')