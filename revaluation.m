%% revaluation
transitionMatrix= [2, 1, 1, 1; ...
                   3, 1, 2, 2; ...
                   3, 2, 7, 4; ...
                   4, 4, 3, 5; ...
                   5, 6, 4, 5; ...
                   5, 6, 6, 6; ...
                   7, 7, 8, 3; ...
                   8, 9, 8, 7; ... 
                   8, 9, 9, 9;];

valueCraisins = 5;
valueM = 1;
vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];

% q-learning parameters
LR=.01;
epsilon=.1;
gamma=.9;

states = size(transitionMatrix,1);
actions = size(transitionMatrix,2);
QvalueMatrix=zeros(states,actions);


numTrials=100;
for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(QvalueMatrix(currentState,:), epsilon);
        newState = transitionMatrix(currentState,action);
        RPE = vectorRewards(newState)+gamma*max(QvalueMatrix(newState,:)) - QvalueMatrix(currentState, action);
        QvalueMatrix(currentState, action) = QvalueMatrix(currentState, action) + LR*(RPE);
        currentState = newState;
    end
end
% currently, when we take the "max" of the action values in the new state , 
% we are greedily choosing the best option. to implement an "on policy" learning algorithm, 
 % we would have a policy for choosing what action to take, then update value based on
 % this choice made by a policy. for example, we can use epsilon greedy as a policy.
%  based on the option chosen by this policy, we can then update the value based on this chosen option. 
% epsilon greedy is used as a policy in this case because it could choose random option instead of choosing highest value option.
 
figure
bar(QvalueMatrix(3,:))
   ylabel('q value')
   set(gca, 'xticklabels', {'up', 'right', 'down'})
   
% state 3 is an important state since the choice the agent makes at this state will
% determine whether the agent goes to the craisins or the M&Ms. our graph demonstrates that at state 3, 
% the agent learned the most value from going up, which leads to state 4. Because state 4
% eventually leads to craisins, which is valued more than M&Ms currently,
% it is expected that our model has learned to value craisins more than M&Ms.


%% part 2
valueCraisins = 0;
valueM = 5;
vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];


numTrials=20;
vectorTerminal = NaN(numTrials,1);
for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(QvalueMatrix(currentState,:), epsilon);
        newState = transitionMatrix(currentState,action);
        RPE = vectorRewards(newState)+gamma*max(QvalueMatrix(newState,:)) - QvalueMatrix(currentState, action);
        QvalueMatrix(currentState, action) = QvalueMatrix(currentState, action) + LR*(RPE);
        currentState = newState;
    end
    vectorTerminal(t) = currentState;
end

propCraisins = length(find(vectorTerminal==6))/numTrials
% No, our mdoel does not adjust quickly to the new reward structure
% in 95% of the trials, the agent still ended up in state 5. Therefore, the
% model did not learn to devalue the craisins from the previous Qvalues and
% adjust to the new reward structure.

%% part 3

V = zeros(states,actions);
valueCraisins = 5;
valueM = 1;
vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];

numTrials=100;
vectorTerminal = NaN(numTrials,1);
for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(V(currentState,:), epsilon);
        newState = transitionMatrix(currentState,action);
        V(currentState, action) = vectorRewards(newState) + max(V(newState,:));
        currentState = newState;
    end
    vectorTerminal(t) = currentState;
end
terminal_count = (vectorTerminal == 6);
to_sum = reshape(terminal_count,[10,numTrials/10]);
sum_cols = sum(to_sum);
figure(1);
plot(1:numTrials/10, sum_cols./10)

figure
bar(V(3,:))
   ylabel('q value')
   set(gca, 'xticklabels', {'up', 'right', 'down'})
 
   
% going up and right are valued highly, and going down is valued slightly (the learned values 
% are approx. proportional to the true value of the craisins and M&Ms). This is different 
% from part 1 because model-based learning takes into account the transition structure of the maze,
% leading to valuing of going down at state 3 proportional to the value of
% the M&Ms.

%% part 4

valueCraisins = 0;
valueM = 1;
vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];

numReplayEvents = 1000;

Values=zeros(states,actions);

numTrials=20;
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
            simRPE = vectorRewards(simState)+gamma*max(Values(simState,:)) - Values(currentState, simAction);
            Values(currentState, simAction) = Values(simState, simAction) + LR*(simRPE);
            currentState = simState;
        end
    end
    vectorTerminal(t) = currentState;
end

propCraisins = length(find(vectorTerminal==6))/numTrials

% propCraisins = .35-.5
% the model quickly adjusts to the new reward structure because in 20
% trials, the agent only chooses the craisins 30-50% of the time compared
% to 95% of the time when there is no simulated replay.

%% forward replay of devalued trajectory

transitionMatrix= [2, 1, 1, 1; ...
                   3, 1, 2, 2; ...
                   3, 2, 7, 4; ...
                   4, 4, 3, 5; ...
                   5, 6, 4, 5; ...
                   5, 6, 6, 6; ...
                   7, 7, 8, 3; ...
                   8, 9, 8, 7; ... 
                   8, 9, 9, 9;];
% q-learning parameters
LR=.01;
epsilon=.1;
gamma=.8;

states = size(transitionMatrix,1);
actions = size(transitionMatrix,2);

valueCraisins = 20;
valueM = 1;
vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];

numReplayEvents = 1000;
ValuesDV=zeros(states,actions);

numTrials=100;
vectorTerminalDV = NaN(numTrials,1);
for t=1:numTrials
    currentState=1;
    while ~(currentState==6 || currentState==9)
        [action] = epsilonGreedy(ValuesDV(currentState,:), epsilon);
     
        newState = transitionMatrix(currentState,action);
        RPE = vectorRewards(newState)+gamma*max(ValuesDV(newState,:)) - ValuesDV(currentState, action);
        ValuesDV(currentState, action) = ValuesDV(currentState, action) + LR*(RPE);
        currentState = newState;
        
        currentStateSim = 1;
        for r=1:numReplayEvents
            % selection of action with a favored probability for the lower
            % reward
            if currentStateSim == 3
                P = 1-[vectorRewards(9), vectorRewards(6)]./(vectorRewards(6)+vectorRewards(9));
                poss_states = [7, 4];
                poss_actions = [3, 4];
                poss_ind = find(rand<cumsum(P),1,'first');
                simAction = poss_actions(poss_ind);
                simState = poss_states(poss_ind);
            else
                [simAction] = epsilonGreedy(ValuesDV(currentStateSim,:), epsilon);
                simState = transitionMatrix(currentStateSim,simAction);
            end
            simRPE = vectorRewards(simState)+gamma*max(ValuesDV(simState,:)) - ValuesDV(currentStateSim, simAction);
            ValuesDV(currentStateSim, simAction) = ValuesDV(simState, simAction) + LR*(simRPE);
            currentStateSim = simState;
            
            if currentStateSim == 6 || currentStateSim == 9   
                currentStateSim = 1;
            end
            
        end
    end
    vectorTerminalDV(t) = currentState;
end
terminal_countDV = (vectorTerminalDV == 6);
to_sumDV = reshape(terminal_countDV,[10,numTrials/10]);
sum_colsDV = sum(to_sumDV);
figure(1);
plot([1:numTrials/10]*10, sum_colsDV./10);

figure(2);
bar(ValuesDV(3,:))
   ylabel('q value')
   set(gca, 'xticklabels', {'up', 'down', 'left', 'right'})
 
   
%propCraisins = length(find(vectorTerminal==6))/numTrials

% propCraisins = .35-.5
% the model quickly adjusts to the new reward structure because in 20
% trials, the agent only chooses the craisins 30-50% of the time compared
% to 95% of the time when there is no simulated replay.



% forward replay of valued trajectory



% transitionMatrix= [2, 1, 1, 1; ...
%                    3, 1, 2, 2; ...
%                    3, 2, 7, 4; ...
%                    4, 4, 3, 5; ...
%                    5, 6, 4, 5; ...
%                    5, 6, 6, 6; ...
%                    7, 7, 8, 3; ...
%                    8, 9, 8, 7; ... 
%                    8, 9, 9, 9;];
% q-learning parameters
% LR=.01;
% epsilon=.1;
% gamma=.8;
% 
% states = size(transitionMatrix,1);
% actions = size(transitionMatrix,2);

% valueCraisins = 20;
% valueM = 1;
% vectorRewards= [0, 0, 0, 0, 0, valueCraisins, 0, 0, valueM];

numReplayEvents = 1000;
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
        
        currentStateSim = 1;
        for r=1:numReplayEvents
            % selection of action with a favored probability for the higher
            % reward
            if currentStateSim == 3
                P = [vectorRewards(9), vectorRewards(6)]./(vectorRewards(6)+vectorRewards(9));
                poss_states = [7, 4];
                poss_actions = [3, 4];
                poss_ind = find(rand<cumsum(P),1,'first');
                simAction = poss_actions(poss_ind);
                simState = poss_states(poss_ind);
            else
                [simAction] = epsilonGreedy(Values(currentStateSim,:), epsilon);
                simState = transitionMatrix(currentStateSim,simAction);
            end
            simRPE = vectorRewards(simState)+gamma*max(Values(simState,:)) - Values(currentStateSim, simAction);
            Values(currentStateSim, simAction) = Values(simState, simAction) + LR*(simRPE);
            currentStateSim = simState;
            
            if currentStateSim == 6 || currentStateSim == 9   
                currentStateSim = 1;
            end
            
        end
    end
    vectorTerminal(t) = currentState;
end
terminal_count = (vectorTerminal == 6);
to_sum = reshape(terminal_count,[10,numTrials/10]);
sum_cols = sum(to_sum);
figure(3);
plot([1:numTrials/10]*10, sum_cols./10);

figure(4);
bar(Values(3,:))
   ylabel('q value')
   set(gca, 'xticklabels', {'up', 'down', 'left', 'right'})

   
figure(5);
plot([1:numTrials/10]*10, sum_colsDV./10);
hold on
plot([1:numTrials/10]*10, sum_cols./10);
legend('DV', 'V')
hold off


%propCraisins = length(find(vectorTerminal==6))/numTrials

% propCraisins = .35-.5
% the model quickly adjusts to the new reward structure because in 20
% trials, the agent only chooses the craisins 30-50% of the time compared
% to 95% of the time when there is no simulated replay.