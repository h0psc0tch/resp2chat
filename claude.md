# Coding style

Keep code coments to a minimum. Add comments where logic is necessarily obscure, confusing or is handling specific edge cases.

Avoid in-line function declarations unless doing so would result in a very complex implementation. Ask if unsure.


# Validating
Ensure all code builds correctly

Ensure all tests are run and pass.

When writing tests:
 - write tests in dedicated test packages (e.g. for package "mypackage" tests should be created in "mypackage_test")
 - use https://pkg.go.dev/github.com/stretchr/testify/assert package where appropriate to validate test outcomes.