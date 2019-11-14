---
title: LeetCode 第2题
date: 2019-11-13 17:16:11
tags: Leetcode
---

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* lres = new ListNode(0);
        ListNode* dummy = lres;
        int v1,v2,val,c=0;
        while (l1 || l2)
        {
            v1 = l1? l1->val : 0;
            v2 = l2? l2->val : 0;
            val = v1 + v2 + c;
            c = val/10;
            val = val%10;
            lres->next = new ListNode(val);
            lres = lres->next;
            l1 = l1? l1->next:l1;
            l2 = l2? l2->next:l2;
        }
        if(c)
            lres->next = new ListNode(c);
        return dummy->next;

    }
};
```