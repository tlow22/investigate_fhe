import time

from heir import compile
from heir.mlir import I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
# 64-bit operands keep the MLIR lowering consistent across the multiply and add.
def func(x: Secret[I64], y: Secret[I64]):
    sum_val = x + y
    diff_val = x - y
    mul_val = x * y
    # Break down the expression to avoid type inference issues during lowering
    product = sum_val * diff_val
    expression = product + mul_val
    return expression

total_start = time.perf_counter()

setup_start = time.perf_counter()
func.setup()
setup_time = time.perf_counter() - setup_start

enc_x_start = time.perf_counter()
enc_x = func.encrypt_x(7)
enc_x_time = time.perf_counter() - enc_x_start

enc_y_start = time.perf_counter()
enc_y = func.encrypt_y(8)
enc_y_time = time.perf_counter() - enc_y_start

eval_start = time.perf_counter()
result_enc = func.eval(enc_x, enc_y)
eval_time = time.perf_counter() - eval_start

decrypt_start = time.perf_counter()
result = func.decrypt_result(result_enc)
decrypt_time = time.perf_counter() - decrypt_start

total_time = time.perf_counter() - total_start

print(
    f"Expected result for `func`: {func.original(7, 8)}, FHE result: {result}"
)
print(
    "Timings: setup={setup:.3f}ms, encrypt_x={enc_x:.3f}ms, "
    "encrypt_y={enc_y:.3f}ms, eval={eval:.3f}ms, decrypt={decrypt:.3f}ms, "
    "total={total:.3f}ms".format(
        setup=setup_time * 1000,
        enc_x=enc_x_time * 1000,
        enc_y=enc_y_time * 1000,
        eval=eval_time * 1000,
        decrypt=decrypt_time * 1000,
        total=total_time * 1000,
    )
)
