'''
A gradient reversal layer.
'''
# pylint: disable=arguments-differ

from typing import Tuple
import torch
import torch.autograd as autograd


#########################################
class __GradRev(autograd.Function): # pylint: disable=invalid-name
    '''
    A gradient reversal layer.
    '''

    #########################################
    @staticmethod
    def forward( # type: ignore
        ctx,
        x: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        '''
        Forward pass. Returns the input as-is.

        :param ctx: Context.
        :param x: The input tensor.
        :param lambda_: The lambda constant of the gradient reversal layer.
        :return: The input as-is.
        '''
        ctx.save_for_backward(torch.tensor(lambda_))
        return x

    #########################################
    @staticmethod
    def backward( # type: ignore
        ctx,
        x_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        '''
        Backward pass. Return the gradient multiplied by negative lambda.

        :param ctx: Context.
        :param x_grad: The incoming gradient.
        :return: A pair matching the arguments for `forward` where the first item is the modified
            gradient of `x` and the second item is None (since `lambda_` does not need a gradient).
        '''
        (lambda_,) = ctx.saved_tensors
        return (-lambda_*x_grad, None)


#########################################
def grad_rev_layer(
    x: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    '''
    Apply a gradient reversal layer

    :param x: The input tensor.
    :param lambda_: The lambda constant of the gradient reversal layer.
    :return: The input as-is.
    '''
    return __GradRev.apply(x, lambda_)
