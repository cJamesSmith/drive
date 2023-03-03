1 - 全道，勉强能跑，done给松一点，误差0.4，0.4, 

```python

obs = self.get_obs()
reward -= obs[0] * 10
reward -= obs[1]
reward -= math.fabs(obs[2]) * 10
reward -= obs[3]
reward += 8


if math.fabs(obs[0]) > 0.6 or 10 > math.fabs(obs[2]) > 5:
    print(
        f"fucked {obs}, {self.my_waypoint[self.nearest_id], self.epi_err_dist / self.frame_step, self.epi_err_yaw / self.frame_step}, l={self.frame_step}"
    )
    done = True
    reward -= 500

```