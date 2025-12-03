# Billing Module
The Billing module automates invoice generation, payment tracking, and refund workflows.
Invoices are usually issued on the 1st of every billing period and are due within 10 days.

Features:
- Automatic pro-rata calculations for plan upgrades.
- Multiple payment gateways supported (Stripe, Razorpay, PayPal).
- Tax compliance for India, EU, and US regions.
- Credit notes and refund support.

If a payment is delayed beyond 15 days, the account may be marked as **Overdue** and restricted until payment is received.