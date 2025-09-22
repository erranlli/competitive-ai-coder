import requests
import json

# Working example using the provided submission
source_code = """# -*- coding: utf-8 -*- 
# @project : 《Atcoder》
# @Author : created by bensonrachel on 2021/10/18
# @File : 28.LCIS.py
# 求 find their longest common increasing subsequence最长公共递增子序列。
def output(w,pre_k):
    if(w != -1):
        output(pre_k[w],pre_k)
        print(b[w-1],end=" ")

def dp_solve():
    dp = [[0]*(m+1) for _ in range(n+1)]
    pre_k = [-1] * (m+1)
    for i in range(1,n+1):
        max_k = 0
        k = -1
        for j in range(1,m+1):
            if a[i-1]!=b[j-1]:
                dp[i][j] = dp[i-1][j]
                if a[i-1]>b[j-1] and max_k < dp[i-1][j] :
                    max_k = dp[i-1][j]
                    k = j
            else:
                dp[i][j] = max_k + 1
                pre_k[j] = k

    ans = 0
    w = 0
    for index,value in enumerate(dp[-1]):
        if(value > ans):
            ans = value
            w = index

    print(ans)
    if(ans):
        output(w,pre_k)

if __name__ == '__main__':
    n = int(input())
    a = [int(i) for i in input().split()]
    m = int(input())
    b = [int(i) for i in input().split()]
    ans = dp_solve()"""

# Piston endpoint (adjust to your setup)
endpoint = "http://localhost:2000"

# Language configuration
extension, piston_language = "py", "python"

# Mock problem data - replace with actual dataset row
problem_data = {
    'time_limit': 2.0,
    'memory_limit': 256.0, 
    'input_mode': 'stdio',
    'generated_checker': None,  # No custom checker for this example
    'official_tests': [
        {
            'input': '4\n1 4 2 5\n4\n7 2 1 5',
            'output': '2\n1 5 '
        }
    ]
}

# Use first test case
test_case = problem_data['official_tests'][0]

payload = {
    "language": piston_language,
    "version": "*",
    "files": [
        {
            "name": f"main.{extension}",
            "content": source_code
        },
        {
            "name": "input.txt",
            "content": test_case['input']
        },
        {
            "name": "correct_output.txt", 
            "content": test_case['output']
        },
        # Add checker.py if needed (none in this example)
        *([{"name": "checker.py", "content": problem_data['generated_checker']}] if problem_data['generated_checker'] else []),
        {
            "name": "grader_config",
            "content": "\n".join(
                f"{key}={value}" for key, value in {
                    "TIME_LIMIT": problem_data['time_limit'],
                    "MEMORY_LIMIT": problem_data['memory_limit'],
                    "INPUT_MODE": problem_data['input_mode']
                }.items()
            )
        }
    ],
    "stdin": test_case['input'],  # Provide input via stdin for stdio mode
    "run_timeout": int(problem_data['time_limit'] * 1000),  # Convert to milliseconds
    "run_memory_limit": int(problem_data['memory_limit'] * 1024 * 1024)  # Convert to bytes
}

try:
    # Execute the code
    result = requests.post(f"{endpoint}/api/v2/execute", json=payload, headers={"Content-Type": "application/json"})
    
    if result.status_code == 200:
        response = result.json()
        print("=== EXECUTION RESULT ===")
        print(json.dumps(response, indent=2))
        
        # Check if execution was successful
        compile_success = 'compile' not in response or response['compile']['code'] == 0
        run_success = response['run']['code'] == 0
        
        if compile_success and run_success:
            print(f"\n=== OUTPUT ===")
            print(response['run']['stdout'])
            
            print(f"\n=== EXPECTED ===") 
            print(test_case['output'])
            
            # Simple output comparison (for problems without custom checkers)
            output_clean = response['run']['stdout'].strip()
            expected_clean = test_case['output'].strip()
            
            if output_clean == expected_clean:
                print("\n✅ TEST PASSED")
            else:
                print("\n⚠️  TEST OUTPUT DIFFERS")
                print("Note: For LCIS problems, multiple valid answers exist")
                print("This would need a custom checker to validate properly")
        else:
            print("\n❌ EXECUTION FAILED")
            if 'compile' in response and response['compile']['code'] != 0:
                print("Compile Error:", response['compile']['stderr'])
            if response['run']['code'] != 0:
                print("Runtime Error:", response['run']['stderr'])
    else:
        print(f"❌ HTTP Error {result.status_code}: {result.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Make sure Piston is running on localhost:2000")
    print("Run: docker-compose up -d api")
except Exception as e:
    print(f"❌ Error: {e}")
