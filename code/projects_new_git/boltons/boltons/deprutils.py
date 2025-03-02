# -*- coding: utf-8 -*-

# Copyright (c) 2013, Mahmoud Hashemi
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * The names of the contributors may not be used to endorse or
#      promote products derived from this software without specific
#      prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Note that DeprecationWarnings are ignored by default in Python
2.7/3.2+, so be sure to either un-ignore them in your code, or run
Python with the -Wd flag.
"""

import sys
from warnings import warn

ModuleType = type(sys)

# todo: only warn once


class DeprecatableModule(ModuleType):
    def __init__(self, module):
        name = module.__name__
        super(DeprecatableModule, self).__init__(name=name)
        self.__dict__.update(module.__dict__)

    def __getattribute__(self, name):
        get_attribute = super(DeprecatableModule, self).__getattribute__
        try:
            depros = get_attribute('_deprecated_members')
        except AttributeError:
            self._deprecated_members = depros = {}
        ret = get_attribute(name)
        message = depros.get(name)
        if message is not None:
            warn(message, DeprecationWarning, stacklevel=2)
        return ret


def deprecate_module_member(mod_name, name, message):
    module = sys.modules[mod_name]
    if not isinstance(module, DeprecatableModule):
        sys.modules[mod_name] = module = DeprecatableModule(module)
    module._deprecated_members[name] = message
    return
