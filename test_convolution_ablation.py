"""Regression checks for the three shift conditions and baseline selection."""
import unittest
from types import SimpleNamespace

import torch
import layers
import multitensor_systems
from tied_convolution import TiedDirectionalConv, tied_directional_shift


class ConvolutionTests(unittest.TestCase):
    def test_solve_metrics(self):
        from analyze_example_wandb import solve_metrics
        metrics = solve_metrics(
            [False, True, False, True, True],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            100,
        )
        self.assertEqual(metrics['first_solve_step'], 1)
        self.assertEqual(metrics['stable_solve_step'], 3)
        self.assertEqual(metrics['wall_clock_to_first_solve_seconds'], 2.0)
        self.assertEqual(metrics['wall_clock_to_stable_solve_seconds'], 4.0)
        self.assertEqual(metrics['estimated_flops_to_first_solve'], 200)
        self.assertEqual(metrics['estimated_flops_to_stable_solve'], 400)
        unsolved = solve_metrics([False, False], [1.0, 2.0], 100)
        self.assertIsNone(unsolved['first_solve_step'])
        self.assertIsNone(unsolved['stable_solve_step'])

    def test_projected_matches_original_at_initialization_and_trains_kernels(self):
        torch.manual_seed(7)
        system = multitensor_systems.MultiTensorSystem(2, 3, 3, 5, None)
        x, weights = system.make_multitensor(), system.make_multitensor()
        active = {(1, 1, 1, 1, 1), (1, 0, 1, 1, 1)}
        for dims in system:
            x[dims] = torch.randn(*system.shape(dims, 8), dtype=torch.float64)
            if tuple(dims) in active:
                weights[dims] = [[torch.randn(8, 4, dtype=torch.float64), None],
                                 [torch.randn(4, 8, dtype=torch.float64), None]]
        masks = torch.randint(0, 2, (2, 3, 5, 2)).double()
        conv = TiedDirectionalConv().double()
        expected = layers.shift(x, weights, masks, post_norm=True)
        actual = layers.tied_conv_shift(x, masks, conv, weights, projected=True)
        pure = layers.tied_conv_shift(x, masks, conv, None, projected=False)
        for dims in system:
            torch.testing.assert_close(actual[dims], expected[dims])
            if tuple(dims) in active:
                torch.testing.assert_close(pure[dims], tied_directional_shift(x[dims], masks, conv))
            else:
                self.assertIs(pure[dims], x[dims])
        sum(actual[d].square().sum() for d in active).backward()
        for parameter in conv.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_selection_excludes_failed_relaxed_and_new_conditions(self):
        from modal_runner import select_successful_baselines
        def run(rid, **changes):
            fields = dict(id=rid, state='finished', summary={'pass_2_correct': True},
                          config=dict(task_name='abc', split='training', multitensor_constraints='strict'))
            fields.update(changes)
            return SimpleNamespace(**fields)
        good = run('good')
        runs = [good, run('failed', state='failed'), run('wrong', summary={'pass_2_correct': False}),
                run('missing', summary={}), run('relaxed', config={**good.config, 'multitensor_constraints': 'relaxed'}),
                run('new', config={**good.config, 'shift_variant': 'pure_conv'})]
        self.assertEqual(select_successful_baselines(runs, 'training'), {'abc': ['good']})
        self.assertEqual(select_successful_baselines(runs, 'evaluation'), {})

    def test_deployed_queue_skips_only_matching_finished_conditions(self):
        from submit_modal_queue import finished_conditions
        base = dict(task_name='abc', split='training', iterations=1500,
                    multitensor_constraints='strict', shift_variant='original')
        runs = [
            SimpleNamespace(state='finished', config=base),
            SimpleNamespace(state='crashed', config={**base, 'shift_variant': 'pure_conv'}),
            SimpleNamespace(state='finished', config={**base, 'iterations': 20,
                                                       'shift_variant': 'projected_conv'}),
        ]
        self.assertEqual(finished_conditions(runs, 'training', 1500), {('abc', 'original')})


if __name__ == '__main__':
    unittest.main()
