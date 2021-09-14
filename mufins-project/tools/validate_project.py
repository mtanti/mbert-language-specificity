#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
Validate the project files.
'''

import os
import argparse
import ast
from typing import List
import mufins


#########################################
def check_init(
    code_path: str,
) -> None:
    '''
    Check for any missing __init__.py files.

    :param code_path: The path to the code files.
    '''
    names = os.listdir(code_path)

    if '__init__.py' not in names:
        raise AssertionError('Missing __init__.py in {}'.format(code_path))

    for name in names:
        new_path = os.path.join(code_path, name)
        if os.path.isdir(new_path) and name not in [
            '__pycache__',
            'mock_dataset',
            'mock_target'
        ]:
            check_init(new_path)


#########################################
def check_docs(
    code_path: str,
    doc_path: str,
) -> None:
    '''
    Check for any missing Sphinx documents together.

    :param code_path: The path to the code files.
    :param doc_path: The path to the Sphinx document files.
    '''
    code_names = os.listdir(code_path)
    doc_names = set(os.listdir(doc_path))
    for name in code_names:
        new_code_path = os.path.join(code_path, name)
        new_doc_path = os.path.join(doc_path, name.replace('.py', '.rst'))

        if os.path.isfile(new_code_path) and name.endswith('.py') and name not in [
            '__init__.py'
        ]:
            if not os.path.isfile(new_doc_path):
                raise AssertionError('Missing doc for {}'.format(new_code_path))

            with open(new_doc_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                namespace = lines[3][16:].split('.')
                expected_namespace = (
                    ['mufins']
                    + new_code_path.replace('.py', '').split(os.path.sep)[-len(namespace)+1:]
                )
                if namespace != expected_namespace:
                    raise AssertionError('Wrong namespace for {}'.format(new_doc_path))

        if os.path.isdir(new_code_path) and name not in [
            '__pycache__',
            'tests',
            'resources'
        ]:
            if name not in doc_names:
                raise AssertionError('Missing doc directory for {}'.format(new_code_path))
            if '{}.rst'.format(name) not in doc_names:
                raise AssertionError('Missing doc for {}'.format(new_code_path))

            check_docs(new_code_path, new_doc_path)


#########################################
def check_docstrings(
    code_path: str,
) -> None:
    '''
    Check for any missing Sphinx documents together.

    :param code_path: The path to the code files.
    '''
    names = os.listdir(code_path)

    for name in names:
        new_path = os.path.join(code_path, name)

        if os.path.isfile(new_path) and name.endswith('.py') and name not in [
            '__init__.py'
        ]:
            with open(new_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            if (
                not isinstance(tree.body[0], ast.Expr)
                or not isinstance(tree.body[0].value, ast.Str)
            ):
                raise AssertionError(
                    'Missing module docstring in file \'{}\'.'.format(new_path)
                )

            def check_code(
                path: str,
                body: List[ast.stmt],
            ) -> None:
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        if (
                            not isinstance(node.body[0], ast.Expr)
                            or not isinstance(node.body[0].value, ast.Str)
                        ):
                            raise AssertionError(
                                'Missing docstring in class \'{}\' on line {} in file \'{}\'.'
                                .format(
                                    node.name, node.lineno, path,
                                )
                            )
                        check_code(path, node.body)

                    if isinstance(node, ast.FunctionDef):
                        name: str = node.name
                        line_num: int = node.lineno
                        args: List[str] = [arg.arg for arg in node.args.args if arg.arg != 'self']
                        assert node.returns is not None
                        has_return = (
                            not isinstance(node.returns, ast.NameConstant)
                            or node.returns.value is not None
                        )
                        if (
                            not isinstance(node.body[0], ast.Expr)
                            or not isinstance(node.body[0].value, ast.Str)
                        ):
                            raise AssertionError(
                                'Missing docstring in function \'{}\' on line {} in file \'{}\'.'
                                .format(
                                    name, line_num, path,
                                )
                            )
                        docstring: str = node.body[0].value.s

                        args_mentioned = list()
                        return_mentioned = False
                        for line in docstring.split('\n'):
                            line = line.strip()
                            if line.startswith(':param '):
                                arg = line.split(' ')[1][:-1]
                                args_mentioned.append(arg)
                            if line.startswith(':return:'):
                                return_mentioned = True
                        if args != args_mentioned:
                            raise AssertionError(
                                'The docstring in function \'{}\' on line {} in file \'{}\' does'
                                ' not match the function\'s arguments. Arguments in docstring but'
                                ' not in function: {}. Arguments in function but not in docstring:'
                                ' {}.'
                                .format(
                                    name, line_num, path,
                                    sorted(set(args_mentioned) - set(args)),
                                    sorted(set(args) - set(args_mentioned)),
                                )
                            )
                        if has_return != return_mentioned:
                            raise AssertionError(
                                'The docstring in function \'{}\' on line {} in file \'{}\' does'
                                ' not match the function\'s return annotation.'
                                .format(name, line_num, path)
                            )

            check_code(new_path, tree.body)

        if os.path.isdir(new_path) and name not in [
            '__pycache__',
        ]:
            check_docstrings(new_path)


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate the project files.'
    )

    parser.parse_args()

    check_init(mufins.path)

    check_docs(
        mufins.path,
        os.path.abspath(os.path.join(mufins.path, '..', 'docs', 'api'))
    )

    check_docstrings(mufins.path)
    check_docstrings(os.path.abspath(os.path.join(mufins.path, '..', 'bin')))
    check_docstrings(os.path.abspath(os.path.join(mufins.path, '..', 'tools')))
