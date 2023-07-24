def max_continuous_sum(arr):
    max_sum = arr[0]  # 현재까지의 최대 연속합
    current_sum = arr[0]  # 현재까지의 연속합

    for num in arr[1:]:
        # 이전까지의 연속합과 현재 숫자를 더한 값과 현재 숫자 중에서 더 큰 값을 선택하여 현재까지의 연속합을 갱신
        current_sum = max(num, current_sum + num)
        # 최대 연속합과 현재까지의 연속합 중에서 더 큰 값을 선택하여 최대 연속합을 갱신
        max_sum = max(max_sum, current_sum)

    return max_sum

def main():
    n = int(input())  # 수열의 길이
    sequence = list(map(int, input().split()))  # 수열
    result = max_continuous_sum(sequence)
    print(result)

main()
