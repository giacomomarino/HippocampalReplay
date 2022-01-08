transitionMatrix= [2, 1, 1, 1; ...
    3, 1, 2, 2; ...
    3, 2, 7, 4; ...
    4, 4, 3, 5; ...
    5, 6, 4, 5; ...
    5, 6, 6, 6; ...
    7, 7, 8, 3; ...
    8, 9, 8, 7; ...
    8, 9, 9, 9;];

valueFood = 3;
valueWater = 3;
vectorRewards= [0, 0, 0, 0, 0, valueFood, 0, 0, valueWater];

LR=.001;
epsilon=.1;
gamma=.2;

states = size(transitionMatrix,1);
actions = size(transitionMatrix,2);
QvalueMatrix=zeros(states,actions);

numReplayEvents = 100;
Values=zeros(states,actions);

numTrials=100;
vectorTerminal = NaN(numTrials,1);

%% Training

Values=zeros(states,actions);

numTrials=100;
vectorTerminal = NaN(numTrials,1);

for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(Values(currentState,:), epsilon);
        newState = transitionMatrix(currentState,action);
        RPE = vectorRewards(newState)+gamma*max(Values(newState,:)) - Values(currentState, action);
        Values(currentState, action) = Values(currentState, action) + LR*(RPE);
        currentState = newState;
        for r=1:numReplayEvents
            simState=randi(states);
            simAction=randi(actions);
            simTransition=transitionMatrix(simState,simAction);
            simRPE = (vectorRewards(simState) + gamma*max(Values(simState,:))) - Values(currentState, simAction);
            Values(currentState, simAction) = Values(simState, simAction) + LR*(simRPE);
             currentState = simState;
        end
    end
    vectorTerminal(t) = currentState;
end

figure(3);
bar(Values(3,:))
title("Valuing Actions from State 3")
ylabel('Q value')
set(gca, 'xticklabels', {'up', 'down', 'left', 'right'})




%% Backwards Replay


QvalueMatrix=zeros(states,actions);

numReplayEvents = 1000;
Values=zeros(states,actions);

valueFood = 2; % state 6
valueWater = 2; % state 9
vectorRewards= [0, 0, 0, 0, 0, valueFood, 0, 0, valueWater];
foodPath = [5, 2; 4, 4; 3, 4];
waterPath = [8, 2; 7, 3; 3, 3];

numTrials=1000;
vectorTerminal = NaN(numTrials,1);
for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(Values(currentState,:), epsilon);
        
        newState = transitionMatrix(currentState,action);
        RPE = vectorRewards(newState)+gamma*max(Values(newState,:)) - Values(currentState, action);
        Values(currentState, action) = Values(currentState, action) + LR*(RPE);
        currentState = newState;
        
        
        for r=1:numReplayEvents
            % selection of action with a favored probability for the higher
            % reward
            P = [vectorRewards(9), vectorRewards(6)]./(vectorRewards(6)+vectorRewards(9));
            poss_states = [9, 6];
            poss_ind = find(rand<cumsum(P),1,'first');
            simState = poss_states(poss_ind);
            if simState == 6
                for i = 1:length(foodPath)
                    movedFrom = transitionMatrix(foodPath(i, 1), foodPath(i, 2));
                    simRPE = vectorRewards(movedFrom)+gamma*max(Values(movedFrom,:)) - Values(foodPath(i, 1), foodPath(i, 2));
                    Values(foodPath(i, 1), foodPath(i, 2)) = Values(foodPath(i, 1), foodPath(i, 2)) + LR*(simRPE);
                end
            else
                for i = 1:length(waterPath)
                    movedFrom = transitionMatrix(waterPath(i, 1), waterPath(i, 2));
                    simRPE = vectorRewards(movedFrom)+gamma*max(Values(movedFrom,:)) - Values(waterPath(i, 1), waterPath(i, 2));
                    Values(waterPath(i, 1), waterPath(i, 2)) = Values(waterPath(i, 1), waterPath(i, 2)) + LR*(simRPE);
                end
                
            end
        end
    end
    vectorTerminal(t) = currentState;
end
terminal_count = (vectorTerminal == 9);
to_sum = reshape(terminal_count,[10,numTrials/10]);
terminal_countFood = (vectorTerminal == 6);
to_sumFood = reshape(terminal_countFood,[10,numTrials/10]);
sum_cols = sum(to_sum);
sum_colsfood = sum(to_sumFood);


figure(1);
bar(Values(3,:))
title("Valuing Actions from State 3")
ylabel('Q value')
set(gca, 'xticklabels', {'up', 'down', 'left', 'right'})


figure(2);
title("Proportion of Terminal States")
xlabel("Trial #")
plot([1:numTrials/10]*10, sum_colsfood./10);
hold on
plot([1:numTrials/10]*10, sum_cols./10);
legend('Proportion Reaching Food (State 9)', 'Proportion Reaching Water (State 6)')
hold off
