# Compilation Errors in dispatch framework

We've put a lot of effort into early-exit static_asserts that will avoid template
error-message hell and hopefully make common errors a little easier to understand.

We succeed for 3/4 cases in creating nice error messages. However, there's not much we can
do for the case where instantiation fails to find a matching overload - those are
one of the common error scenarios, and in that case, the output is still pretty awful.


## Error Test Compile Launch

```
ctest -R DispatchCompileFail --output-on-failure -V
UpdateCTestConfiguration  from :src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch/DartConfiguration.tcl
Test project src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
```

## DispatchCompileFail_MixedAxisTypes

Common error situation in which user creates an axis with values of different types.

```
test 11
    Start 11: DispatchCompileFail_MixedAxisTypes

11: Test command: miniforge3/envs/fvdb/bin/cmake "--build" "src/fvdb-core/build/cp312-cp312-linux_x86_64-Release" "--target" "DispatchCompileFail_MixedAxisTypes_obj" "--config" "Release"
11: Working Directory: src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch
11: Test timeout computed to be: 10000000
11: [1/1] Building CXX object src/dispatch/CMakeFiles/DispatchCompileFail_MixedAxisTypes_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
11: FAILED: [code=1] src/dispatch/CMakeFiles/DispatchCompileFail_MixedAxisTypes_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
11: miniforge3/envs/fvdb/bin/x86_64-conda-linux-gnu-c++ -DTEST_MIXED_AXIS_TYPES -Isrc/fvdb-core/src/dispatch -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem miniforge3/envs/fvdb/include  -Iminiforge3/envs/fvdb/targets/x86_64-linux/include -O3 -DNDEBUG -std=gnu++20 -MD -MT src/dispatch/CMakeFiles/DispatchCompileFail_MixedAxisTypes_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -MF src/dispatch/CMakeFiles/DispatchCompileFail_MixedAxisTypes_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o.d -o src/dispatch/CMakeFiles/DispatchCompileFail_MixedAxisTypes_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -c src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp
11: In file included from src/fvdb-core/src/dispatch/dispatch/detail.h:7,
11:                  from src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:7:
11: src/fvdb-core/src/dispatch/dispatch/types.h: In instantiation of 'struct dispatch::axis<1, 2.0e+0f>':
11: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:15:15:   required from here
11: src/fvdb-core/src/dispatch/dispatch/types.h:111:25: error: static assertion failed: axis values must be the same type
11:   111 |     static_assert((std::is_same_v<value_type, decltype(V)> && ... && true),
11:       |                    ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
11: src/fvdb-core/src/dispatch/dispatch/types.h:111:25: note: 'std::is_same_v<int, float>' evaluates to false
11: ninja: build stopped: subcommand failed.
1/4 Test #11: DispatchCompileFail_MixedAxisTypes ......   Passed    0.52 sec
```

## DispatchCompileFail_SubspaceNotWith

Common error case in which a requested instantiation subspace in the creation of a dispatch
table is not within the full space.

```
test 12
    Start 12: DispatchCompileFail_SubspaceNotWithin

12: Test command: miniforge3/envs/fvdb/bin/cmake "--build" "src/fvdb-core/build/cp312-cp312-linux_x86_64-Release" "--target" "DispatchCompileFail_SubspaceNotWithin_obj" "--config" "Release"
12: Working Directory: src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch
12: Test timeout computed to be: 10000000
12: [1/1] Building CXX object src/dispatch/CMakeFiles/DispatchCompileFail_SubspaceNotWithin_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
12: FAILED: [code=1] src/dispatch/CMakeFiles/DispatchCompileFail_SubspaceNotWithin_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
12: miniforge3/envs/fvdb/bin/x86_64-conda-linux-gnu-c++ -DTEST_SUBSPACE_NOT_WITHIN -Isrc/fvdb-core/src/dispatch -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem miniforge3/envs/fvdb/include  -Iminiforge3/envs/fvdb/targets/x86_64-linux/include -O3 -DNDEBUG -std=gnu++20 -MD -MT src/dispatch/CMakeFiles/DispatchCompileFail_SubspaceNotWithin_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -MF src/dispatch/CMakeFiles/DispatchCompileFail_SubspaceNotWithin_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o.d -o src/dispatch/CMakeFiles/DispatchCompileFail_SubspaceNotWithin_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -c src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp
12: In file included from src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:8:
12: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h: In instantiation of 'dispatch::dispatch_table<dispatch::axes<Axes ...>, ReturnType(Args ...)>::dispatch_table(Factory&&, Subs ...) [with Factory = main()::<lambda(auto:22)>; Subs = {dispatch::axes<dispatch::axis<1, 99> >}; Axes = {dispatch::axis<1, 2, 3>}; ReturnType = int; Args = {}]':
12: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:23:72:   required from here
12: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h:50:58: error: static assertion failed: Subs must be within the axes
12:    50 |         static_assert((within<Subs, axes_type> && ... && true), "Subs must be within the axes");
12:       |                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~
12: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h:50:58: note: constraints not satisfied
12: In file included from src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:7:
12: src/fvdb-core/src/dispatch/dispatch/detail.h:213:9:   required by the constraints of 'template<class Sub, class Full> concept dispatch::within'
12: src/fvdb-core/src/dispatch/dispatch/detail.h:213:40: note: the expression '(is_within_v<Sub, Full>)() [with Sub = dispatch::axes<dispatch::axis<1, 99> >; Full = dispatch::axes<dispatch::axis<1, 2, 3> >]' evaluated to 'false'
12:   213 | concept within = is_within_v<Sub, Full>();
12:       |                  ~~~~~~~~~~~~~~~~~~~~~~^~
12: ninja: build stopped: subcommand failed.
2/4 Test #12: DispatchCompileFail_SubspaceNotWithin ...   Passed    0.56 sec
```


## DispatchCompileFail_WrongTupleType

Common situation where the value types in a permutation tuple used to dispatch at runtime are
the wrong types. If they're values that aren't instantiated, that's a valid runtime scenario and
an exception will be thrown. However, wrong types will not compile, and those errors are clean.

```
test 14
    Start 14: DispatchCompileFail_WrongTupleType

14: Test command: miniforge3/envs/fvdb/bin/cmake "--build" "src/fvdb-core/build/cp312-cp312-linux_x86_64-Release" "--target" "DispatchCompileFail_WrongTupleType_obj" "--config" "Release"
14: Working Directory: src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch
14: Test timeout computed to be: 10000000
14: [1/1] Building CXX object src/dispatch/CMakeFiles/DispatchCompileFail_WrongTupleType_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
14: FAILED: [code=1] src/dispatch/CMakeFiles/DispatchCompileFail_WrongTupleType_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
14: miniforge3/envs/fvdb/bin/x86_64-conda-linux-gnu-c++ -DTEST_WRONG_TUPLE_TYPE -Isrc/fvdb-core/src/dispatch -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem miniforge3/envs/fvdb/include  -Iminiforge3/envs/fvdb/targets/x86_64-linux/include -O3 -DNDEBUG -std=gnu++20 -MD -MT src/dispatch/CMakeFiles/DispatchCompileFail_WrongTupleType_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -MF src/dispatch/CMakeFiles/DispatchCompileFail_WrongTupleType_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o.d -o src/dispatch/CMakeFiles/DispatchCompileFail_WrongTupleType_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -c src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp
14: In file included from src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:7:
14: src/fvdb-core/src/dispatch/dispatch/detail.h: In instantiation of 'constexpr std::optional<long unsigned int> dispatch::linear_index_from_value_tuple(axes<SubAxis>, const std::tuple<T0>&) [with Axis = axis<1, 2>; T0 = float]':
14: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:42:58:   required from here
14: src/fvdb-core/src/dispatch/dispatch/detail.h:674:24: error: static assertion failed: value type mismatch
14:   674 |     static_assert(std::is_same_v<v_type, axis_value_type>, "value type mismatch");
14:       |                   ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
14: src/fvdb-core/src/dispatch/dispatch/detail.h:674:24: note: 'std::is_same_v<float, int>' evaluates to false
14: src/fvdb-core/src/dispatch/dispatch/detail.h: In instantiation of 'constexpr std::optional<long unsigned int> dispatch::index_of_value(axis<V0, V ...>, auto:14) [with auto V0 = 1; auto ...Vs = {2}; auto:14 = float]':
14: src/fvdb-core/src/dispatch/dispatch/detail.h:675:26:   required from here
14: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:42:58:   in 'constexpr' expansion of 'dispatch::linear_index_from_value_tuple<axis<1, 2>, float>((Axes(), Axes()), std::make_tuple(_Elements&& ...) [with _Elements = {float}]())'
14: src/fvdb-core/src/dispatch/dispatch/detail.h:462:24: error: static assertion failed: value type mismatch
14:   462 |     static_assert(std::is_same_v<v_type, a_v_type>, "value type mismatch");
14:       |                   ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
14: src/fvdb-core/src/dispatch/dispatch/detail.h:462:24: note: 'std::is_same_v<float, int>' evaluates to false
14: ninja: build stopped: subcommand failed.
4/4 Test #14: DispatchCompileFail_WrongTupleType ......   Passed    0.55 sec

The following tests passed:
        DispatchCompileFail_MixedAxisTypes
        DispatchCompileFail_SubspaceNotWithin
        DispatchCompileFail_OpMissingOverload
        DispatchCompileFail_WrongTupleType

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   2.21 sec
```



## DispatchCompileFail_OpMissingOverload

Common error case in which there's no overload found, as either a free function or
via the "op" methods of an Op struct, which correspond to a particular permutation that
is requested when the table is constructed.

This one we can't really help with, because there's no place for us to attach to, to place the
static_assert error message. This is no different than regular instantiation though, we
haven't made it worse in any way.

```
test 13
    Start 13: DispatchCompileFail_OpMissingOverload

13: Test command: miniforge3/envs/fvdb/bin/cmake "--build" "src/fvdb-core/build/cp312-cp312-linux_x86_64-Release" "--target" "DispatchCompileFail_OpMissingOverload_obj" "--config" "Release"
13: Working Directory: src/fvdb-core/build/cp312-cp312-linux_x86_64-Release/src/dispatch
13: Test timeout computed to be: 10000000
13: [1/1] Building CXX object src/dispatch/CMakeFiles/DispatchCompileFail_OpMissingOverload_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
13: FAILED: [code=1] src/dispatch/CMakeFiles/DispatchCompileFail_OpMissingOverload_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o
13: miniforge3/envs/fvdb/bin/x86_64-conda-linux-gnu-c++ -DTEST_OP_MISSING_OVERLOAD -Isrc/fvdb-core/src/dispatch -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem miniforge3/envs/fvdb/include  -Iminiforge3/envs/fvdb/targets/x86_64-linux/include -O3 -DNDEBUG -std=gnu++20 -MD -MT src/dispatch/CMakeFiles/DispatchCompileFail_OpMissingOverload_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -MF src/dispatch/CMakeFiles/DispatchCompileFail_OpMissingOverload_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o.d -o src/dispatch/CMakeFiles/DispatchCompileFail_OpMissingOverload_obj.dir/tests/compile_errors/dispatch_compile_errors.cpp.o -c src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp
13: In file included from src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:8:
13: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h: In instantiation of 'static dispatch::dispatch_table<dispatch::axes<Axes ...>, ReturnType(Args ...)>::return_type dispatch::dispatch_table<dispatch::axes<Axes ...>, ReturnType(Args ...)>::op_call(Args ...) [with Op = main()::IncompleteOp; Coord = dispatch::tag<2>; Axes = {dispatch::axis<1, 2>}; ReturnType = int; Args = {int}; return_type = int]':
13: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h:99:20:   required from 'dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> [with auto:20 = dispatch::tag<2>; dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::function_pointer_type = int (*)(int)]'
13: src/fvdb-core/src/dispatch/dispatch/axes_map.h:262:57:   required from 'void dispatch::detail::create_and_store_visitor<Axes, T, Factory>::operator()(Tag) const [with Tag = dispatch::tag<2>; Axes = dispatch::axes<dispatch::axis<1, 2> >; T = int (*)(int); Factory = dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)>]'
13: miniforge3/envs/fvdb/lib/gcc/x86_64-conda-linux-gnu/13.4.0/include/c++/bits/invoke.h:61:36:   required from 'constexpr _Res std::__invoke_impl(__invoke_other, _Fn&&, _Args&& ...) [with _Res = void; _Fn = dispatch::detail::create_and_store_visitor<dispatch::axes<dispatch::axis<1, 2> >, int (*)(int), dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> >&; _Args = {dispatch::tag<2>}]'
13: miniforge3/envs/fvdb/lib/gcc/x86_64-conda-linux-gnu/13.4.0/include/c++/bits/invoke.h:96:40:   required from 'constexpr typename std::__invoke_result<_Functor, _ArgTypes>::type std::__invoke(_Callable&&, _Args&& ...) [with _Callable = dispatch::detail::create_and_store_visitor<dispatch::axes<dispatch::axis<1, 2> >, int (*)(int), dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> >&; _Args = {dispatch::tag<2>}; typename __invoke_result<_Functor, _ArgTypes>::type = void]'
13: miniforge3/envs/fvdb/lib/gcc/x86_64-conda-linux-gnu/13.4.0/include/c++/functional:113:27:   required from 'constexpr std::invoke_result_t<_Fn, _Args ...> std::invoke(_Callable&&, _Args&& ...) [with _Callable = dispatch::detail::create_and_store_visitor<dispatch::axes<dispatch::axis<1, 2> >, int (*)(int), dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> >&; _Args = {dispatch::tag<2>}; invoke_result_t<_Fn, _Args ...> = void]'
13: src/fvdb-core/src/dispatch/dispatch/visit_spaces.h:88:21:   required from 'static void dispatch::detail::axes_visit_helper<Axes, std::integer_sequence<long unsigned int, linearIndices ...> >::visit(Visitor&) [with Visitor = dispatch::detail::create_and_store_visitor<dispatch::axes<dispatch::axis<1, 2> >, int (*)(int), dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> >; Axes = dispatch::axes<dispatch::axis<1, 2> >; long unsigned int ...linearIndices = {0, 1}]'
13: src/fvdb-core/src/dispatch/dispatch/visit_spaces.h:103:87:   required from 'void dispatch::visit_axes_space(Visitor&, Axes) [with Visitor = detail::create_and_store_visitor<axes<axis<1, 2> >, int (*)(int), dispatch_table<axes<axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)> >; Axes = axes<axis<1, 2> >]'
13: src/fvdb-core/src/dispatch/dispatch/axes_map.h:274:21:   required from 'void dispatch::detail::create_and_store_helper(dispatch::axes_map<Axes, T>&, Factory&, dispatch::axes<AxisTypes ...>) [with Axes = dispatch::axes<dispatch::axis<1, 2> >; T = int (*)(int); Factory = dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)>; SubAxes = {dispatch::axis<1, 2>}; dispatch::axes_map<Axes, T> = std::unordered_map<dispatch::axes_map_key<dispatch::axes<dispatch::axis<1, 2> > >, int (*)(int), dispatch::axes_map_hash<dispatch::axes<dispatch::axis<1, 2> > >, dispatch::axes_map_equal<dispatch::axes<dispatch::axis<1, 2> > >, std::allocator<std::pair<const dispatch::axes_map_key<dispatch::axes<dispatch::axis<1, 2> > >, int (*)(int)> > >]'
13: src/fvdb-core/src/dispatch/dispatch/axes_map.h:303:37:   required from 'void dispatch::create_and_store(axes_map<Axes, T>&, Factory&, Subs ...) [with Axes = axes<axis<1, 2> >; T = int (*)(int); Factory = dispatch_table<axes<axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)>; Subs = {axes<axis<1, 2> >}; axes_map<Axes, T> = std::unordered_map<axes_map_key<axes<axis<1, 2> > >, int (*)(int), axes_map_hash<axes<axis<1, 2> > >, axes_map_equal<axes<axis<1, 2> > >, std::allocator<std::pair<const axes_map_key<axes<axis<1, 2> > >, int (*)(int)> > >]'
13: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h:54:25:   required from 'dispatch::dispatch_table<dispatch::axes<Axes ...>, ReturnType(Args ...)>::dispatch_table(Factory&&, Subs ...) [with Factory = dispatch::dispatch_table<dispatch::axes<dispatch::axis<1, 2> >, int(int)>::from_op<main()::IncompleteOp>()::<lambda(auto:20)>; Subs = {dispatch::axes<dispatch::axis<1, 2> >}; Axes = {dispatch::axis<1, 2>}; ReturnType = int; Args = {int}]'
13: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:36:55:   required from here
13: src/fvdb-core/src/dispatch/dispatch/dispatch_table.h:123:22: error: cannot convert 'tag<'nontype_argument_pack' not supported by dump_expr<expression error>>' to 'tag<'nontype_argument_pack' not supported by dump_expr<expression error>>'
13:   123 |         return Op::op(Coord{}, args...);
13:       |                ~~~~~~^~~~~~~~~~~~~~~~~~
13: src/fvdb-core/src/dispatch/tests/compile_errors/dispatch_compile_errors.cpp:30:12: note:   initializing argument 1 of 'static int main()::IncompleteOp::op(dispatch::tag<1>, int)'
13:    30 |         op(dispatch::tag<1>, int x) {
13:       |            ^~~~~~~~~~~~~~~~
13: ninja: build stopped: subcommand failed.
3/4 Test #13: DispatchCompileFail_OpMissingOverload ...   Passed    0.57 sec
```
