import torch


class Gomoku:
    """
    play1: 1,
    play2: -1,
    empty: 0,
    """

    def __init__(
        self, _size=5, _win_num=4, _player=1, _terminal=None, _history=None, _board=None
    ):
        self.size = _size
        self.win_num = _win_num
        self.num_actions = _size * _size

        self.player = _player  # cur player, 1 or -1
        self.terminal = _terminal  # 1, -1 or 0 for draw, None for unknown
        self.history = [] if _history is None else _history
        self.board = torch.zeros(self.size, self.size) if _board is None else _board

    def clone(self):
        return Gomoku(
            self.size,
            self.win_num,
            self.player,
            self.terminal,
            self.history[:],
            self.board.clone(),
        )

    def get_input(self):
        ac_1 = torch.zeros(self.size, self.size)
        try:
            x, y = self._action2loc(self.history[-1])
            ac_1[x, y] = 1
        except IndexError:
            pass
        ac_2 = torch.zeros(self.size, self.size)
        try:
            x, y = self._action2loc(self.history[-2])
            ac_2[x, y] = 1
        except IndexError:
            pass
        return torch.stack([self.board * self.player, ac_1, ac_2])

    def get_mask(self):
        return (self.board == 0).flatten()

    def _check_horizon(self, _value, _x, _y):
        left = 0
        y_left = _y - 1
        while y_left >= 0 and self.board[_x, y_left] == _value:
            left += 1
            y_left -= 1
        right = 0
        y_right = _y + 1
        while y_right < self.size and self.board[_x, y_right] == _value:
            right += 1
            y_right += 1
        if left + right + 1 >= self.win_num:  # horizon win
            self.terminal = _value

        return self.terminal

    def _check_vertical(self, _value, _x, _y):
        up = 0
        x_up = _x - 1
        while x_up >= 0 and self.board[x_up, _y] == _value:
            up += 1
            x_up -= 1
        down = 0
        x_down = _x + 1
        while x_down < self.size and self.board[x_down, _y] == _value:
            down += 1
            x_down += 1
        if up + down + 1 >= self.win_num:  # vertical win
            self.terminal = _value

        return self.terminal

    def _check_inv_slash(self, _value, _x, _y):
        up_left = 0
        x_up = _x - 1
        y_left = _y - 1
        while x_up >= 0 and y_left >= 0 and self.board[x_up, y_left] == _value:
            up_left += 1
            x_up -= 1
            y_left -= 1
        down_right = 0
        x_down = _x + 1
        y_right = _y + 1
        while (
            x_down < self.size
            and y_right < self.size
            and self.board[x_down, y_right] == _value
        ):
            down_right += 1
            x_down += 1
            y_right += 1
        if up_left + down_right + 1 >= self.win_num:  # inv slash win
            self.terminal = _value

        return self.terminal

    def _check_slash(self, _value, _x, _y):
        up_right = 0
        x_up = _x - 1
        y_right = _y + 1
        while x_up >= 0 and y_right < self.size and self.board[x_up, y_right] == _value:
            up_right += 1
            x_up -= 1
            y_right += 1
        down_left = 0
        x_down = _x + 1
        y_left = _y - 1
        while (
            x_down < self.size and y_left >= 0 and self.board[x_down, y_left] == _value
        ):
            down_left += 1
            x_down += 1
            y_left -= 1
        if up_right + down_left + 1 >= self.win_num:  # slash win
            self.terminal = _value

        return self.terminal

    def _action2loc(self, _action: int):
        # get loc
        return _action // self.size, _action % self.size

    def action(self, _action: int):
        x, y = self._action2loc(_action)
        # update board and player
        assert self.board[x, y] == 0 and self.terminal is None
        value = self.player
        self.board[x, y] = value
        self.history.append(_action)
        self.player *= -1
        # check terminal
        if self._check_horizon(value, x, y) is not None:
            return self.terminal
        if self._check_vertical(value, x, y) is not None:
            return self.terminal
        if self._check_inv_slash(value, x, y) is not None:
            return self.terminal
        if self._check_slash(value, x, y) is not None:
            return self.terminal
        # check draw
        if len(self.history) == self.size * self.size:
            self.terminal = 0
        return self.terminal

    def __repr__(self):
        return (
            f"Cur Player: {self.player}\n"
            f"History: {self.history}\n"
            f"Terminal: {self.terminal}\n"
            f"Board:\n"
            f"{self.board}"
        )

    def human_self(self):
        print(self)
        while self.terminal is None:
            x = input("Please input x: ")
            y = input("Please input y: ")
            action = int(x) * self.size + int(y)
            self.action(action)
            print(self)

        print(f"!!Result!!: {self.terminal}")


if __name__ == "__main__":
    gomoku = Gomoku()
    gomoku.human_self()
