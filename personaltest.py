#!/usr/bin/env python3
"""
Test Query Set Evaluation Script
"""

from search_engine import evaluate
import time

# Test query set
queries = [
    "“第一所”和“第一支”",
    "中国人民大学经济学院教授对马克思主义政治经济学的研究",
    "读史读经典征文",
    "2023年高瓴人工智能学院的吴玉章奖学金获得者",
    "“一站式”社区",
    "沙盘心理咨询",
    "机器学习中的隐私保护技术",
    "六校思源计划涉及的是哪六个学校？",
    "社会发展与管理大数据中心成立时间",
    "人工智能产生的就业变革",
    "刘伟和陈彦斌的研究", 
    "总书记关于教育的专题研讨会",
    "2025青年创新汇客厅第9期",
    "哲学社会科学的“五路大军”",
    "人大在哪个博物馆开展了思政课？",
    "招募“先锋闯将”辅导员"
]

# Standard answers
answers = [
    ("http://xsc.ruc.edu.cn/info/1022/5292.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/Hzxyj/18329e494eb6484192eda6fc6d184b17.htm",),
    ("http://xsc.ruc.edu.cn/info/1021/1616.htm",),
    ("http://xsc.ruc.edu.cn/info/1022/5131.htm", "http://xsc.ruc.edu.cn/info/1022/5125.htm"),
    ("http://xsc.ruc.edu.cn/yzs_sq.htm", "http://xsc.ruc.edu.cn/info/1022/5224.htm", "http://xsc.ruc.edu.cn/info/1021/5124.htm", "http://xsc.ruc.edu.cn/info/1021/5176.htm"),
    ("http://xsc.ruc.edu.cn/xlzx1.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/kydt/kyxm/63d45f85be194fd1af4a648e92162cdd.htm",),
    ("http://xsc.ruc.edu.cn/info/1022/4167.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/kyjg/kxyyjjg/index.htm", "http://keyan.ruc.edu.cn/kyjg/kxyyjjg/index.htm"),
    ("http://keyan.ruc.edu.cn/wwsy/kydt/xsjl/ed0bbec9cc004ec0af0deafe6adbab86.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/Hzxyj/53a45aea5f47434cb194e359edde0772.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/kydt/xsjl/90d6cd45838347b8893d574f9491563c.htm", "http://keyan.ruc.edu.cn/wwsy/dqgz/413d41ec8c474dfbb056f731b31104ea.htm"),
    ("http://keyan.ruc.edu.cn/wwsy/kydt/xsjl/ce85f5cad5514cc8af8fc229da924f1e.htm",),
    ("http://keyan.ruc.edu.cn/wwsy/kydt/xsjl/fd932c6d0ee6428897500fd14cfbc67d.htm",),
    ("http://xsc.ruc.edu.cn/info/1022/5272.htm",),
    ("http://xsc.ruc.edu.cn/info/1022/5195.htm",)
]

def calculate_metrics(retrieved_urls, expected_urls):
    """
    Calculate evaluation metrics
    
    Args:
        retrieved_urls: List of retrieved URLs
        expected_urls: Tuple of expected URLs
    
    Returns:
        dict: Dictionary containing various metrics
    """
    if not expected_urls:
        return {"precision": 0, "recall": 0, "f1": 0, "hit": False, "rr": 0}
    
    # Convert to sets for set operations
    retrieved_set = set(retrieved_urls)
    expected_set = set(expected_urls)
    
    # Calculate intersection
    intersection = retrieved_set & expected_set
    
    # Precision = Number of relevant documents retrieved / Total number of documents retrieved
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
    
    # Recall = Number of relevant documents retrieved / Total number of relevant documents
    recall = len(intersection) / len(expected_set) if expected_set else 0
    
    # F1 score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Hit rate = Whether at least one relevant document is found
    hit = len(intersection) > 0
    
    # Calculate Reciprocal Rank
    rr = 0
    for rank, url in enumerate(retrieved_urls, 1):
        if url in expected_set:
            rr = 1 / rank
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "found": len(intersection),
        "expected": len(expected_set),
        "retrieved": len(retrieved_set),
        "rr": rr
    }

def test_search_modes():
    """Test different search modes"""
    modes = [
        ("BM25", {"enable_dedup": False, "use_sdm": False}),
        ("BM25+Dedup", {"enable_dedup": True, "use_sdm": False}),
        ("BM25+SDM", {"enable_dedup": False, "use_sdm": True}),
        ("BM25+SDM+Dedup", {"enable_dedup": True, "use_sdm": True}),
    ]
    
    print("=" * 140)
    print(f"{'Query':<40} {'Mode':<15} {'Hit':<6} {'Found/Exp':<10} {'P':<8} {'R':<8} {'F1':<8} {'RR':<8}")
    print("=" * 140)
    
    mode_stats = {mode_name: {"hits": 0, "total_precision": 0, "total_recall": 0, "total_f1": 0, "total_rr": 0} 
                  for mode_name, _ in modes}
    
    for i, (query, expected) in enumerate(zip(queries, answers)):
        print(f"\nQuery {i+1}: {query[:35]}{'...' if len(query) > 35 else ''}")
        
        for mode_name, params in modes:
            start_time = time.time()
            try:
                results = evaluate(query, **params)
                search_time = time.time() - start_time
                
                # 计算指标
                metrics = calculate_metrics(results, expected)
                
                # Update statistics
                stats = mode_stats[mode_name]
                if metrics["hit"]:
                    stats["hits"] += 1
                stats["total_precision"] += metrics["precision"]
                stats["total_recall"] += metrics["recall"]
                stats["total_f1"] += metrics["f1"]
                stats["total_rr"] += metrics["rr"]  # 添加RR统计
                
                # Display results
                hit_symbol = "✓" if metrics["hit"] else "✗"
                found_expected = f"{metrics['found']}/{metrics['expected']}"
                
                print(f"{'':42} {mode_name:<15} {hit_symbol:<6} {found_expected:<10} "
                      f"{metrics['precision']:<8.3f} {metrics['recall']:<8.3f} {metrics['f1']:<8.3f} "
                      f"{metrics['rr']:<8.3f} ({search_time:.2f}s)")
                
                # Display found URLs (first 3 only)
                if results and metrics["hit"]:
                    for j, url in enumerate(results[:3]):
                        if url in expected:
                            print(f"{'':42} {'':15} ✓ {url}")
                        else:
                            print(f"{'':42} {'':15} - {url}")
                
            except Exception as e:
                print(f"{'':42} {mode_name:<15} ERROR: {str(e)}")
    
    # Display overall statistics
    print("\n" + "=" * 140)
    print("Overall Evaluation Results:")
    print("=" * 140)
    print(f"{'Mode':<15} {'Hit Rate':<10} {'MRR@20':<10} {'Avg P':<10} {'Avg R':<10} {'Avg F1':<10}")
    print("-" * 80)
    
    total_queries = len([q for q in queries if answers[queries.index(q)]])  # Exclude queries without answers
    
    for mode_name, stats in mode_stats.items():
        hit_rate = stats["hits"] / total_queries if total_queries > 0 else 0
        mrr = stats["total_rr"] / total_queries if total_queries > 0 else 0  # Calculate MRR@20
        avg_precision = stats["total_precision"] / total_queries if total_queries > 0 else 0
        avg_recall = stats["total_recall"] / total_queries if total_queries > 0 else 0
        avg_f1 = stats["total_f1"] / total_queries if total_queries > 0 else 0
        
        print(f"{mode_name:<15} {hit_rate:<10.3f} {mrr:<10.3f} {avg_precision:<10.3f} {avg_recall:<10.3f} {avg_f1:<10.3f}")

def evaluate_like_client():
    """Simulate client.py evaluation method"""
    print("=" * 80)
    print("CLIENT Mode Evaluation (Simulating Server Evaluation)")
    print("=" * 80)
    
    total_rr = 0
    total_queries = 0
    
    for i, (query, expected) in enumerate(zip(queries, answers)):
        if not expected:  # 跳过没有答案的查询
            continue
            
        print(f"\nProcessing query {i+1}/{len(queries)}: 「{query[:50]}{'...' if len(query) > 50 else ''}」")
        
        # Use the same configuration as client.py
        # results = evaluate(query, enable_dedup=False, use_sdm=True)
        results = evaluate(query)
        print(f"Returned {len(results)} results (using BM25+SDM)")
        
        # Calculate reciprocal rank
        rr = 0
        for rank, url in enumerate(results, 1):
            if url in expected:
                rr = 1 / rank
                print(f"  ✓ First relevant document rank: {rank} (RR={rr:.3f})")
                print(f"    {url}")
                break
        else:
            print("  ✗ No relevant documents found (RR=0.000)")
        
        total_rr += rr
        total_queries += 1
    
    mrr = total_rr / total_queries if total_queries > 0 else 0
    print(f"\n" + "=" * 80)
    print(f"Final MRR@20 score: {mrr:.4f}")
    print(f"Processed queries: {total_queries}")
    print("=" * 80)

def test_single_query(query_idx=0):
    """Detailed test of a single query"""
    if query_idx >= len(queries):
        print(f"Query index {query_idx} out of range")
        return
    
    query = queries[query_idx]
    expected = answers[query_idx] if query_idx < len(answers) else ()
    
    print(f"Detailed test query: {query}")
    print(f"Expected results: {len(expected)} URLs")
    for i, url in enumerate(expected, 1):
        print(f"  {i}. {url}")
    
    print("\n" + "-" * 80)
    
    # Test BM25
    results_bm25 = evaluate(query, enable_dedup=False, use_sdm=False)
    metrics_bm25 = calculate_metrics(results_bm25, expected)
    
    print(f"\nBM25 results (found {len(results_bm25)}):")
    print(f"Precision: {metrics_bm25['precision']:.3f}, Recall: {metrics_bm25['recall']:.3f}, F1: {metrics_bm25['f1']:.3f}")
    
    for i, url in enumerate(results_bm25[:10], 1):
        status = "✓" if url in expected else " "
        print(f"  {status} {i:2d}. {url}")
    
    # Test BM25+SDM (if available)
    try:
        results_sdm = evaluate(query, enable_dedup=False, use_sdm=True)
        metrics_sdm = calculate_metrics(results_sdm, expected)
        
        print(f"\nBM25+SDM results (found {len(results_sdm)}):")
        print(f"Precision: {metrics_sdm['precision']:.3f}, Recall: {metrics_sdm['recall']:.3f}, F1: {metrics_sdm['f1']:.3f}")
        
        for i, url in enumerate(results_sdm[:10], 1):
            status = "✓" if url in expected else " "
            print(f"  {status} {i:2d}. {url}")
            
    except Exception as e:
        print(f"\nBM25+SDM test failed: {e}")

if __name__ == '__main__':
    print("RUC IR System - Query Set Evaluation")
    print("=" * 120)
    
    # Select test mode
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            query_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            test_single_query(query_idx)
        elif sys.argv[1] == "client":
            # Simulate client.py evaluation method
            evaluate_like_client()
        else:
            print("Usage: python test.py [single] [client] [query_index]")
    else:
        # Run evaluation for all queries
        test_search_modes()