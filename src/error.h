/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <stdexcept>
#include <iostream>
#include <sstream>

#define throw_error(message) \
    do { \
        std::cerr << message << std::endl; throw std::runtime_error((std::stringstream("") << message).str()); \
    } while (false)
