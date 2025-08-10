def test_import():
    import gym_tetris
    from gym_tetris import TetrisEnv
    assert callable(TetrisEnv)
