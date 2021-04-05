import random

import gym
import numpy as np

# This is the process_frame function we implemented earlier
from process_frame import process_frame


class GameWrapper:
    """Wrapper for the environment provided by Gym"""
    # 생성자, gym과 인자를 통한 객체 생성
    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = 4

        self.state = None
        self.last_lives = 0

    def reset(self, evaluation=False):
        """Resets the environment : 리셋, 과거 env를 리셋한다.
        Arguments:
            evaluation: agent가 평가될 때 True로 설정, no-op steps의 임의의 수를 설정한다
        """

        self.frame = self.env.reset()
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps. This adds an element of randomness, so that the each evaluation is slightly different.
        # 임의의 no-op steps의 번호를 받는다. 이는 원소에 random함을 추가하여 각 평가가 조금씩 다르게 만들어 준다
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # 첫 상태에서 첫 frame을 4회 쌓는다
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    def step(self, action, render_mode=None):
        """Performs an action and observes the result : 행동을 취하고 결과를 확인한다.
        Arguments:
            action: agent가 선택한 행동의 번호 
            render_mode: 'None'이라면 아무것도 render하지 않는다. 'human'이면 새 스크린에 render하고, 'rgb_array'라면 rgb 값의 np.array를 리턴한다.
        Returns:
            processed_frame: 행동의 결과로 연산된 새 프레임
            reward: 행동으로 인한 보상
            terminal: game이 종료되었는지를 표시하는 값
            life_lost: life가 깎였는지 표시하는 값
            new_frame: 행동의 결과로 연산된 새 프레임(raw frame, 가공되지 않음)
            render_mode가 'rgb_array'로 함수가 실행되었다면, 렌더링된 rgb_array도 리턴한다. 
        """
        new_frame, reward, terminal, info = self.env.step(action)

        # env.setp(action)이 리턴하는 잘 쓰이지 않는 info나 meta 데이터에서 agent가 가진 life의 수를 알 수 있고
        # 이를 통해 우리는 life_lost를 판별할 수 있다.

        # 이 life_lost를 이용하여 agent가 아무것도 안하는걸 방지하고 게임을 시작하도록 강제할 수 있다.
        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        # 이전에 정의한 process_frame함수를 통해 frame을 연산하고 state에 붙인다.
        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        # return_mode에 따라 리턴형태를 달리한다. 
        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost