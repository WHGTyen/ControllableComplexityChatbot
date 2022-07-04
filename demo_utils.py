from parlai.core.agents import Agent
from parlai.core.opt import Opt
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.worlds import create_task

def start_interaction(agent: Agent, opt: Opt):
    """
    Starts interactive mode. Code adapted from parlai/scripts/interactive.py
    """
    human_agent = LocalHumanAgent(opt)
    world_logger = WorldLogger(opt)
    world = create_task(opt, [human_agent, agent])

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()
        if world.epoch_done() or world.get_total_parleys() <= 0:
            # chat was reset with [DONE], [EXIT] or EOF
            if world_logger is not None:
                world_logger.reset()
            continue

        if world_logger is not None:
            world_logger.log(world)
        if opt.get('display_examples'):
            print("---")
            print(world.display())

    if world_logger is not None:
        # dump world acts to file
        world_logger.write(opt['outfile'], world, file_format=opt['save_format'])

