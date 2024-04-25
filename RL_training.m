robotParametersRL
mdl = "rlWalkingBipedRobot";
open_system(mdl)

numObs = 29;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "observations";

numAct = 6;
actInfo = rlNumericSpec([numAct 1],LowerLimit=-1,UpperLimit=1);
actInfo.Name = "foot_torque";

blk = mdl + "/RL Agent";
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
env.ResetFcn = @(in) walkerResetFcn(in, ...
    upper_leg_length/100, ...
    lower_leg_length/100, ...
    h/100);

AgentSelection = "DDPG";
switch AgentSelection
    case "DDPG"
        agent = DDPGAgent(numObs,obsInfo,numAct,actInfo,Ts);
    case "TD3"
        agent = TD3Agent(numObs,obsInfo,numAct,actInfo,Ts);
    otherwise
        disp("Assign AgentSelection to DDPG or TD3")
end     

maxEpisodes = 3000;
maxSteps = floor(Tf/Ts);
trainOpts = rlTrainingOptions(MaxEpisodes=maxEpisodes,MaxStepsPerEpisode=maxSteps,ScoreAveragingWindowLength=250,Verbose=false,Plots="training-progress",StopTrainingCriteria="EpisodeCount",StopTrainingValue=maxEpisodes);

trainOpts.UseParallel = false;
trainOpts.ParallelizationOptions.Mode = "async";

doTraining = false;
simin = [0 0 0; 5 3 0; 10 5 3];
if doTraining    
    trainingStats = train(agent,env,trainOpts);
    save run1
else
    if strcmp(AgentSelection,"DDPG")
       load(fullfile("run4_6000.mat"),"agent")
    else
       load(fullfile("run2.mat"),"agent")
    end  
end

rng(0)
simOptions = rlSimulationOptions(MaxSteps=maxSteps);
experience = sim(env,agent,simOptions);