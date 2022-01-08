%% Simulating just TD Learning


maze = [8, 7, 3, 4, 5; 9, NaN, 2, NaN, 6; NaN, NaN, 1, NaN, NaN];

% Setting up movement matrix
% rows = states, columns equal action (order: left, up, right, down)
stateMoves = [1, 2, 1, 1; 2, 3, 2, 1; 7, 3, 4, 2; 3, 4, 5, 4; 4, 5, 5, 6; ...
    6, 5, 6, 6; 8, 7, 3, 7; 8, 8, 7, 9; 9, 8, 9, 9];


alpha = .01;
epsilon = .5;
DF = .9;

qVals = zeros(9, 4);



rewards = zeros(9, 1);
rewards(6) = 1; % food
rewards(9) = 5; % water

for i = 1:100

    currentState = 1;
    j = 1;
    while 1
        %disp(currentState)
        % chose action
        chosenAction = epsilonGreedy(qVals(currentState, :), epsilon);
        % new state from transition matrix
        choices(j) = chosenAction;
        newState = stateMoves(currentState, chosenAction);
        RPE = rewards(newState) + (DF*max(qVals(newState, :))) - ...
            qVals(currentState, chosenAction);
        qVals(currentState, chosenAction) = qVals(currentState, chosenAction)...
            + alpha*RPE;
        currentState = newState;
        j = j + 1;
        if currentState == 6
            break
        elseif currentState == 9
            break
        end
            
    end
end

% visualize state 3:
bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})


%% Including the Model-Based Planning

alpha = .01;
epsilon = .2;
DF = .9;

qVals = zeros(9, 4);

rewards = zeros(9, 1);
rewards(6) = 1; % food
rewards(9) = 5; % waterz
states = NaN(100, 1);

for i = 1:100

    currentState = 1;
    j = 1;
    while 1
        % chose action
        chosenAction = epsilonGreedy(qVals(currentState, :), epsilon);
        % new state from transition matrix
        choices(j) = chosenAction;
        states(j) = currentState;
        newState = stateMoves(currentState, chosenAction);
        moveProb = 1;
        if newState == currentState
            moveProb = 0;
        elseif (j > 1)
            if states(j) == states(j - 1)
            moveProb = 0;
            end
        end
        [m, ~] = max(qVals(newState, :));
        qVals(currentState, chosenAction) = ...
            qVals(currentState, chosenAction) + ...
            alpha*moveProb*(DF*m + rewards(newState));
        currentState = newState;
        if currentState == 6
            break
        elseif currentState == 9
            break
        end
        j = j + 1;
            
    end
end

% visualize state 3:
bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})



%% Training with Random Replay

alpha = .01;
epsilon = .2;
DF = .2;

qVals = zeros(9, 4);

rewards = zeros(9, 1);
rewards(6) = 3; % food
rewards(9) = 3; % water


numReplayEvents = 1000;

for i = 1:100

    currentState = 1;
    j = 2;
    while 1
        %disp(currentState)
        % chose action
        if currentState == 6
            break
        elseif currentState == 9
            break
        end
        % chose action
        chosenAction = epsilonGreedy(qVals(currentState, :), epsilon);
        % new state from transition matrix
        choices(j) = chosenAction;
        newState = stateMoves(currentState, chosenAction);
        states(j) = currentState;
        moveProb = 1;
        if newState == currentState
            moveProb = 0;
        end    
        [m, ~] = max(qVals(newState, :));
        RPE = moveProb*(DF*m + rewards(newState)) - qVals(currentState, chosenAction);
        
        if choices(j) == choices(j-1)
            RPE = 0;
        end    
        qVals(currentState, chosenAction) = qVals(currentState, chosenAction) + alpha*RPE;

        currentState = newState;
        for j = 1:numReplayEvents
            randState = randi(9);
            randAction = randi(4);
            randMove = stateMoves(randState, randAction);
            [m, ~] = max(qVals(randMove, :));
            moveProb = 1;
            if randState == randMove
                moveProb = 0;
            end
            RPE = moveProb*(DF*m + rewards(randMove)) - qVals(randState, randAction);
            qVals(randState, randAction) = qVals(randState, randAction) + alpha*RPE;
  
        end
        j = j + 1;
            
    end
end

% visualize state 3:
bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})





%% Implement Replay

alpha = .01;
epsilon = .2;
DF = .7;

qVals = zeros(9, 4);
rewards = zeros(9, 1);
rewards(6) = 3; % food
rewards(9) = 3; % water


numReplayEvents = 1000;

for i = 1:2:100
    % alternate food-restricted and water-restricted states
    if mod(i, 2) == 0 %water restricted
       rewards(6) = 0;
       rewards(9) = 6;
    elseif mod(i, 2) == 1 %food restricted
       rewards(9) = 3; 
       rewards(6) = 1;
    end
        
    
    currentState = 1;
    j = 2;
    while 1
        currentStateSim = currentState;
        for j = 1:numReplayEvents
            if currentStateSim == 3 && rewards(9) ~= rewards(6)
               if rewards(9) > rewards(6)
                   newStateSim = 4;
               elseif rewards(9) < rewards(6)
                   newStateSim = 7;
               end
            else
                chosenAction = epsilonGreedy(qVals(currentStateSim, :), epsilon);
                newStateSim = stateMoves(currentStateSim, chosenAction);
            end
            [m, ~] = max(qVals(newStateSim, :));
            moveProb = 1;
            if newStateSim == currentStateSim
                moveProb = 0;
            end    
            RPE = moveProb*(DF*m + rewards(newStateSim)) - qVals(currentStateSim, chosenAction);
            qVals(currentStateSim, chosenAction) = qVals(currentStateSim, chosenAction) + RPE*alpha;
            currentStateSim = newStateSim;
            if currentStateSim == 6
                currentStateSim = 1;
            elseif currentStateSim == 9   
                currentStateSim = 1;
            end
        end
        % chose action
        if currentState == 6
            break
        elseif currentState == 9
            break
        end
        % chose action
        chosenAction = epsilonGreedy(qVals(currentState, :), epsilon);
        % new state from transition matrix
        choices(j) = chosenAction;
        newState = stateMoves(currentState, chosenAction);
        states(j) = currentState;
        moveProb = 1;
        if newState == currentState
            moveProb = 0;
        end    
        [m, ~] = max(qVals(newState, :));
        RPE = moveProb*(DF*m + rewards(newState)) - qVals(currentState, chosenAction);
        
        if choices(j) == choices(j-1)
            RPE = 0;
        end
        qVals(currentState, chosenAction) = qVals(currentState, chosenAction) + alpha*RPE;

        currentState = newState;
        j = j + 1;       
    end
end

% visualize state 3:
bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})

%% Salo playfun

alpha = .01;
epsilon = .2;
DF = .5;


rewards = zeros(9, 1);
rewards(6) = 3; % food
rewards(9) = 3; % water


numReplayEvents = 1000;

for i = 1:2:100
    % alternate food-restricted and water-restricted states
    if mod(i, 2) == 1 %water restricted
       rewards(9) = 6;
       rewards(6) = 0;
    elseif mod(i, 2) == 0 %food restricted
       rewards(6) = 6;
    end
    
    currentState = 1;
    j = 2;
    while 1
        currentStateSim = currentState;
        for j = 1:numReplayEvents
            if currentStateSim == 3
               if rewards(9) > rewards(6)
                   newStateSim = 4;
                   chosenAction = 1;
               else
                   newStateSim = 7;
                   chosenAction = 3;
               end
            else
                chosenAction = epsilonGreedy(qVals(currentStateSim, :), epsilon);
                newStateSim = stateMoves(currentStateSim, chosenAction);
            end
            [m, ~] = max(qVals(newStateSim, :));
            moveProb = 1;
            if newStateSim == currentStateSim
                moveProb = 0;
            end
            RPE = moveProb*(DF*m + rewards(newStateSim)) - qVals(currentStateSim, chosenAction);
            qVals(currentStateSim, chosenAction) = qVals(currentStateSim, chosenAction) + RPE*alpha;
            currentStateSim = newStateSim;
            if currentStateSim == 6
                currentStateSim = 1;
            elseif currentStateSim == 9   
                currentStateSim = 1;
            end
        end
        % chose action
        if currentState == 6
            break
        elseif currentState == 9
            break
        end
        % chose action
        chosenAction = epsilonGreedy(qVals(currentState, :), epsilon);
        % new state from transition matrix
        choices(j) = chosenAction;
        newState = stateMoves(currentState, chosenAction);
        states(j) = currentState;
        moveProb = 1;
        if newState == currentState
            moveProb = 0;
        end    
        [m, ~] = max(qVals(newState, :));
        RPE = moveProb*(DF*m + rewards(newState)) - qVals(currentState, chosenAction);
        
        qVals(currentState, chosenAction) = qVals(currentState, chosenAction) + alpha*RPE;

        currentState = newState;
        j = j + 1;
        
    end
end

% visualize state 3:
bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})

%%

rewards = zeros(9, 1);
rewards(6) = 3; % food
rewards(9) = 1; % water
qVals = zeros(9, 4);
DF = .9;
epsilon = .05;

currentStateSim = 1;
for j = 1:numReplayEvents
    if currentStateSim == 3 && rewards(9) ~= rewards(6)
        if rewards(9) > rewards(6)
            newStateSim = 4;
        elseif rewards(9) < rewards(6)
            newStateSim = 7;
        end
    else
        chosenAction = epsilonGreedy(qVals(currentStateSim, :), epsilon);
        newStateSim = stateMoves(currentStateSim, chosenAction);
    end
    [m, ~] = max(qVals(newStateSim, :));
    RPE = (DF*m + rewards(newStateSim)) - qVals(currentStateSim, chosenAction);
    qVals(currentStateSim, chosenAction) = qVals(currentStateSim, chosenAction) + RPE*alpha;
    currentStateSim = newStateSim;
    if currentStateSim == 6
        disp(6)
        currentStateSim = 1;
    elseif currentStateSim == 9
        disp(9)
        currentStateSim = 1;
    end
end

bar(qVals(3, :))
ylabel('q-value')
set(gca, 'xticklabels', {'left', 'up', 'right', 'down'})

%%
for i = 1:100
    chosenAction = epsilonGreedy(zeros(4, 1), epsilon);
    choice(i) = chosenAction
end