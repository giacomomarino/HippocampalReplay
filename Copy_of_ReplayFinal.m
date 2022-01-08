%% Training -- learn values
TrainedValues = training(20, 10);
figure;
bar(TrainedValues(3,:))
ylabel('Q-Value')
title('State 3 Q-Values (Training)')
set(gca, 'xticklabels', {'Up', 'Down', 'Left', 'Right'})
 
%% Replay -- LR=.01; epsilon=.3; gamma=.7;
for lr = [0.1, .01]
   for epsilon = [.1, .3]
        for gamma = [.6, .8]
            plot_title = strcat('LR = ', string(lr), ', epsilon = ', string(epsilon), ', gamma = ', string(gamma));
            show = (lr == 0.1 && epsilon ==.1 && gamma == 0.8);
            replayPlots(lr,epsilon,gamma, TrainedValues, plot_title, show)
        end
   end
end