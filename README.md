Final submission of JelDor team, FLATland challenge 2019

A few words about suggested approach:

1) In spite of this was a reinforcement learning challenge we decided to use operational approach.
Actually classical multi-agent path planning algorithms still are able to compete against ML solutions which I hope was prooved by the results of this submission.

2) The main idea of the algorithm was to use priority multi-agent approach (when agents build their path one-by-one, considering reservations of antecedent ones)
with improvements in malfunction tracking, design of start schedule building etc. That means that we always have full complete paths for an every single agent - 
in case there won`t be any malfunctions later.

3) The most notable part of a project was malfunction processing. When an agents gets a malfunction it`s path become invalid so we need to rebuild it.
However it may cause some conflicts if already built paths are intersected somewhere.
My approach is to rebuild only those path which need to be changed (as the result of conflicts or malfunction) - it saves a lot of time and makes
the algorithm as fast as ML approaches are (and better than some of them).

4) Unfortunatelly, time limits don`t allow to perform any complete algorithms like "push and rotate" so sometimes my solution is unable to build new path after malfunction occurence.
That is the reason why the solution got only 95% of done agents (instead of 97-98% if there are no simulation fails).