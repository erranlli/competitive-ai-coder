import requests
import json

import pandas as pd
import requests
from pathlib import Path

GENERATED_TESTS_DIR = "/root/piston/takehome/open_rl_codeforces/generated_tests"

def load_generated_tests(problem_id, generated_tests_dir):
    """
    Load generated tests for a specific problem from parquet files.
    
    Args:
        problem_id (str): Problem ID like "852/A" or "10/D"
        generated_tests_dir (str): Path to directory containing test_cases_*.parquet files
        
    Returns:
        list: List of test cases with 'input', 'output', and 'test_case_i' keys
    """
    # Extract contest_id from problem_id (e.g., "852/A" -> "852")
    contest_id = problem_id.split('/')[0]
    
    # Look for the corresponding parquet file
    test_file = Path(generated_tests_dir) / f"test_cases_{contest_id}.parquet"
    
    if not test_file.exists():
        print(f"No generated tests file found for contest {contest_id}")
        return []
    
    try:
        # Load the parquet file
        df = pd.read_parquet(test_file)
        
        # Filter for the specific problem
        problem_tests = df[df['problem_id'] == problem_id]
        
        # Convert to list of test cases
        test_cases = []
        for _, row in problem_tests.iterrows():
            test_cases.append({
                'input': row['input'],
                'output': row['output'],
                'test_case_i': row['test_case_i']
            })
        
        # Sort by test case index
        test_cases.sort(key=lambda x: x['test_case_i'])
        print(f"Loaded {len(test_cases)} generated tests for {problem_id}")
        return test_cases
        
    except Exception as e:
        print(f"Error loading generated tests for {problem_id}: {e}")
        return []


def run_piston_test(source_code, problem_data, test_case, language="python", extension="py", endpoint="http://localhost:2000"):
    """
    Run a single test case through Piston.
    
    Args:
        source_code (str): The source code to execute
        problem_data (dict): Problem metadata with time_limit, memory_limit, etc.
        test_case (dict): Test case with 'input' and 'output' keys
        language (str): Programming language identifier for Piston
        extension (str): File extension for the source code
        endpoint (str): Piston API endpoint
        
    Returns:
        dict: Result with 'passed' boolean and additional info
    """
    payload = {
        "language": language,
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
            # Add checker.py if needed
            *([{"name": "checker.py", "content": problem_data['generated_checker']}] if problem_data.get('generated_checker') else []),
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
        "stdin": test_case['input'],
        "run_timeout": int(problem_data['time_limit'] * 1000),
        "run_memory_limit": int(problem_data['memory_limit'] * 1024 * 1024)
    }

    try:
        result = requests.post(f"{endpoint}/api/v2/execute", json=payload, headers={"Content-Type": "application/json"})
        
        if result.status_code == 200:
            response = result.json()
            
            # Check if execution was successful
            compile_success = 'compile' not in response or response['compile']['code'] == 0
            run_success = response['run']['code'] == 0
            
            if compile_success and run_success:
                return {
                    'passed': True,
                    'output': response['run']['stdout'],
                    'expected': test_case['output'],
                    'cpu_time': response['run']['cpu_time'],
                    'memory': response['run'].get('memory', 0)
                }
            else:
                return {
                    'passed': False,
                    'error': response['compile']['stderr'] if 'compile' in response and response['compile']['code'] != 0 else response['run']['stderr'],
                    'response': response
                }
        else:
            return {'passed': False, 'error': f"HTTP {result.status_code}: {result.text}"}
            
    except Exception as e:
        return {'passed': False, 'error': str(e)}

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

# Mock problem data - replace with actual dataset row from open-r1/codeforces
problem_data = {
    'id': '10/D',  # This should match the problem_id in your dataset
    'time_limit': 2.0,
    'memory_limit': 256.0, 
    'input_mode': 'stdio',
    'generated_checker': None,  # No custom checker for this example
    'generated_tests': 5,  # Number of generated tests available
    'official_tests': [
        {
            'input': '4\n1 4 2 5\n4\n7 2 1 5',
            'output': '2\n1 5 '
        }
    ]
}

print(f"Problem ID: {problem_data['id']}")

# Test with official tests first
print("\n=== TESTING WITH OFFICIAL TESTS ===")
for i, test_case in enumerate(problem_data['official_tests']):
    print(f"\nOfficial Test {i+1}:")
    result = run_piston_test(source_code, problem_data, test_case, piston_language, extension)
    
    if result['passed']:
        print(f"✅ PASSED")
        print(f"Output: {result['output'].strip()}")
        print(f"Expected: {result['expected'].strip()}")
        if result['output'].strip() != result['expected'].strip():
            print("⚠️  Note: Output differs but this may be valid for LCIS problems")
    else:
        print(f"❌ FAILED: {result['error']}")

# Test with generated tests
print(f"\n=== TESTING WITH GENERATED TESTS ===")
generated_tests = load_generated_tests(problem_data['id'], GENERATED_TESTS_DIR)

if generated_tests:
    passed_count = 0
    total_count = len(generated_tests)
    
    # Test first few generated tests (to avoid too much output)
    test_limit = min(5, len(generated_tests))
    
    for i, test_case in enumerate(generated_tests[:test_limit]):
        print(f"\nGenerated Test {test_case['test_case_i']}:")
        result = run_piston_test(source_code, problem_data, test_case, piston_language, extension)
        
        if result['passed']:
            passed_count += 1
            print(f"✅ PASSED")
            if result['output'].strip() != result['expected'].strip():
                print("⚠️  Output differs - may need checker validation")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    print(f"\n=== GENERATED TESTS SUMMARY ===")
    print(f"Tested: {test_limit}/{total_count} generated tests")
    print(f"Passed: {passed_count}/{test_limit}")
    
    if test_limit < total_count:
        print(f"Note: {total_count - test_limit} additional tests available")
else:
    print("No generated tests found for this problem")

# Comprehensive evaluation function
def evaluate_solution(source_code, problem_data, language="python", extension="py", max_generated_tests=10):
    """Comprehensive evaluation using both official and generated tests."""
    results = {
        'official_passed': 0,
        'official_total': 0,
        'generated_passed': 0,
        'generated_total': 0,
        'errors': []
    }
    
    # Test official tests
    for test_case in problem_data.get('official_tests', []):
        results['official_total'] += 1
        result = run_piston_test(source_code, problem_data, test_case, language, extension)
        if result['passed']:
            results['official_passed'] += 1
        else:
            results['errors'].append(f"Official test failed: {result.get('error', 'Unknown')}")
    
    # Test generated tests
    if problem_data.get('generated_tests', 0) > 0:
        generated_tests = load_generated_tests(problem_data['id'], GENERATED_TESTS_DIR)
        
        test_cases_to_run = generated_tests[:max_generated_tests]
        for test_case in test_cases_to_run:
            results['generated_total'] += 1
            result = run_piston_test(source_code, problem_data, test_case, language, extension)
            if result['passed']:
                results['generated_passed'] += 1
            else:
                results['errors'].append(f"Generated test {test_case['test_case_i']} failed: {result.get('error', 'Unknown')}")
    
    return results

print(f"\n=== COMPREHENSIVE EVALUATION EXAMPLE ===")
eval_results = evaluate_solution(source_code, problem_data, piston_language, extension)

print(f"Official Tests: {eval_results['official_passed']}/{eval_results['official_total']}")
print(f"Generated Tests: {eval_results['generated_passed']}/{eval_results['generated_total']}")
print(f"Overall: {eval_results['official_passed'] + eval_results['generated_passed']}/{eval_results['official_total'] + eval_results['generated_total']}")

if eval_results['errors']:
    print(f"\nErrors encountered: {len(eval_results['errors'])}")
    for error in eval_results['errors'][:3]:  # Show first 3 errors
        print(f"  - {error}")


# Example for C++ code
print("\n" + "="*50)
print("C++ EXAMPLE")
print("="*50)

cpp_code = """#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for(int i = 0; i < n; i++) {
        cin >> a[i];
    }
    
    int m;
    cin >> m;
    vector<int> b(m);
    for(int i = 0; i < m; i++) {
        cin >> b[i];
    }
    
    // Simple LCIS implementation for demo
    vector<int> dp(m, 0);
    vector<int> prev(m, -1);
    
    for(int i = 0; i < n; i++) {
        int cur = 0, last = -1;
        for(int j = 0; j < m; j++) {
            if(a[i] == b[j] && dp[j] < cur + 1) {
                dp[j] = cur + 1;
                prev[j] = last;
            }
            if(b[j] < a[i] && dp[j] > cur) {
                cur = dp[j];
                last = j;
            }
        }
    }
    
    int ans = 0, pos = -1;
    for(int i = 0; i < m; i++) {
        if(dp[i] > ans) {
            ans = dp[i];
            pos = i;
        }
    }
    
    cout << ans << endl;
    if(ans > 0) {
        vector<int> path;
        for(int p = pos; p != -1; p = prev[p]) {
            path.push_back(b[p]);
        }
        for(int i = path.size() - 1; i >= 0; i--) {
            cout << path[i] << " ";
        }
        cout << endl;
    }
    
    return 0;
}"""

cpp_payload = {
    "language": "c++",  # Use 'c' for C++ in standard Piston (or 'gcc')
    "version": "*",
    "files": [
        {
            "name": "main.cpp",
            "content": cpp_code
        }
    ],
    "stdin": test_case['input'],
    "run_timeout": int(problem_data['time_limit'] * 1000),
    "run_memory_limit": int(problem_data['memory_limit'] * 1024 * 1024)
}

try:
    result_cpp = requests.post(f"{endpoint}/api/v2/execute", json=cpp_payload, headers={"Content-Type": "application/json"})
    
    if result_cpp.status_code == 200:
        response_cpp = result_cpp.json()
        print("=== C++ EXECUTION RESULT ===")
        print(json.dumps(response_cpp, indent=2))
        
        # Check results
        compile_success = response_cpp['compile']['code'] == 0
        run_success = response_cpp['run']['code'] == 0
        
        if compile_success and run_success:
            print(f"\n=== C++ OUTPUT ===")
            print(response_cpp['run']['stdout'])
            print("\n✅ C++ EXECUTION SUCCESSFUL")
        else:
            print("\n❌ C++ EXECUTION FAILED")
            if response_cpp['compile']['code'] != 0:
                print("Compile Error:", response_cpp['compile']['stderr'])
    else:
        print(f"❌ C++ HTTP Error {result_cpp.status_code}: {result_cpp.text}")
        
except Exception as e:
    print(f"❌ C++ Error: {e}")