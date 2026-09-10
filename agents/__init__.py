from agents.c51 import C51Agent
from agents.codac import CODACAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iqn import IQNAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.value_flows import ValueFlowsAgent
from agents.mav_flows_per_agent import MAVFlowPerAgentAgent
from agents.mav_flows import MAVFlowAgent
from agents.mav_flows_continuous import MAVFlowContinuousAgent

agents = dict(
    c51=C51Agent,
    codac=CODACAgent,
    fbrac=FBRACAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iqn=IQNAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    value_flows=ValueFlowsAgent,
    mav_flow = MAVFlowAgent,
    mav_flow_per_agent = MAVFlowPerAgentAgent,
    mav_flow_continuous=MAVFlowContinuousAgent,
)
