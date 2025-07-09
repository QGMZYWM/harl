"""HASAC algorithm."""
import torch
from harl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from harl.models.policy_models.transformer_policy import TransformerEnhancedPolicy
from harl.utils.discrete_util import gumbel_softmax
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class HASAC(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__
        
        # Check if Transformer enhancement is enabled
        self.use_transformer = args.get("use_transformer", False)
        self.use_contrastive_learning = args.get("use_contrastive_learning", False)

        if self.use_transformer:
            # Use Transformer-enhanced policy
            self.actor = TransformerEnhancedPolicy(args, obs_space, act_space, device)
        elif act_space.__class__.__name__ == "Box":
            self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
        else:
            self.actor = StochasticMlpPolicy(args, obs_space, act_space, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # Store previous embeddings for contrastive learning
        self.previous_contrastive_info = None
        
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True, agent_id=None):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
            agent_id: (int) agent identifier for Transformer history tracking
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        
        if self.use_transformer:
            # Use Transformer-enhanced policy
            actions, _, _, _, contrastive_info = self.actor(
                obs, agent_id=agent_id, available_actions=available_actions, 
                deterministic=not stochastic
            )
            # Store contrastive info for potential loss computation
            self.previous_contrastive_info = contrastive_info
        elif self.action_type == "Box":
            actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        else:
            actions = self.actor(obs, available_actions, stochastic)
        return actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True, agent_id=None):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
            agent_id: (int) agent identifier for Transformer history tracking
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
            contrastive_info: (dict) information for contrastive learning (if using Transformer)
        """
        obs = check(obs).to(**self.tpdv)
        contrastive_info = None
        
        if self.use_transformer:
            # Use Transformer-enhanced policy
            actions, logp_actions, _, _, contrastive_info = self.actor(
                obs, agent_id=agent_id, available_actions=available_actions, 
                deterministic=not stochastic
            )
            # Store contrastive info for loss computation
            self.previous_contrastive_info = contrastive_info
        elif self.action_type == "Box":
            actions, logp_actions = self.actor(
                obs, stochastic=stochastic, with_logprob=True
            )
        elif self.action_type == "Discrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = gumbel_softmax(
                logits, hard=True, device=self.device
            )  # onehot actions
            logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        elif self.action_type == "MultiDiscrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = []
            logp_actions = []
            for logit in logits:
                action = gumbel_softmax(
                    logit, hard=True, device=self.device
                )  # onehot actions
                logp_action = torch.sum(action * logit, dim=-1, keepdim=True)
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(logp_actions, dim=-1)
        
        if self.use_transformer:
            return actions, logp_actions, contrastive_info
        else:
            return actions, logp_actions

    def compute_contrastive_loss(self, contrastive_info=None):
        """Compute contrastive learning loss for Transformer-enhanced policy.
        Args:
            contrastive_info: (dict) contrastive learning information from actor
        Returns:
            contrastive_loss: (torch.Tensor) contrastive learning loss
        """
        if not self.use_contrastive_learning or not self.use_transformer:
            return torch.tensor(0.0, device=self.device)
        
        if contrastive_info is None:
            contrastive_info = self.previous_contrastive_info
        
        if contrastive_info is None:
            return torch.tensor(0.0, device=self.device)
        
        return self.actor.compute_contrastive_loss(contrastive_info)
    
    def reset_history(self, agent_id=None):
        """Reset history buffers for Transformer-enhanced policy.
        Args:
            agent_id: (int) agent identifier, if None reset all agents
        """
        if self.use_transformer:
            self.actor.reset_history(agent_id)

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
