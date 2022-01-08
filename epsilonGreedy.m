function [chosenAction] = epsilonGreedy(Qvals, epsilon)

epsilonProb = 1 - epsilon;

if ~all(Qvals)
    chosenAction = randi(4);
elseif epsilonProb > rand(1)
    [~, index] = max(Qvals);
    chosenAction = index;
else
    chosenAction = randi(4);
end

