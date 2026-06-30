import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const PlasticApp());

class PlasticApp extends StatelessWidget {
  const PlasticApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '플라스틱 분류기',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _ipController = TextEditingController(text: '192.168.0.100');
  bool _loading = false;
  String? _result;
  Map<String, dynamic>? _probabilities;
  String? _error;

  Color _colorForLabel(String label) {
    switch (label) {
      case 'PET':  return Colors.blue;
      case 'PP':   return Colors.green;
      case 'HDPE': return Colors.orange;
      case 'LDPE': return Colors.purple;
      default:     return Colors.teal;
    }
  }

  Future<void> _predict() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) {
      setState(() => _error = 'IP 주소를 입력하세요');
      return;
    }

    setState(() {
      _loading = true;
      _result = null;
      _probabilities = null;
      _error = null;
    });

    try {
      final response = await http
          .post(
            Uri.parse('http://$ip:5000/predict'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _result = data['prediction'] as String;
          _probabilities = data['probabilities'] as Map<String, dynamic>;
        });
      } else {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() => _error = data['error']?.toString() ?? '서버 오류 ${response.statusCode}');
      }
    } catch (e) {
      setState(() => _error = '연결 실패 — IP를 확인하세요\n$e');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _ipController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text('플라스틱 분류기'),
        centerTitle: true,
        backgroundColor: Colors.teal,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // IP 입력 카드
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '라즈베리파이 설정',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: Colors.grey),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: _ipController,
                      decoration: const InputDecoration(
                        labelText: 'IP 주소',
                        hintText: '예: 192.168.0.100',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.router, color: Colors.teal),
                        isDense: true,
                      ),
                      keyboardType: TextInputType.number,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // 측정 버튼
            ElevatedButton.icon(
              onPressed: _loading ? null : _predict,
              icon: Icon(_loading ? Icons.hourglass_top : Icons.science, size: 26),
              label: Text(
                _loading ? '측정 중...' : '플라스틱 측정 시작',
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 18),
                backgroundColor: Colors.teal,
                foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.teal.shade200,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
            const SizedBox(height: 16),

            // 로딩바
            if (_loading) ...[
              const LinearProgressIndicator(color: Colors.teal),
              const SizedBox(height: 8),
              const Text(
                '센서가 측정하고 있습니다. 잠시 기다려주세요...',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey),
              ),
              const SizedBox(height: 16),
            ],

            // 오류
            if (_error != null)
              Card(
                color: Colors.red.shade50,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(_error!, style: const TextStyle(color: Colors.red)),
                      ),
                    ],
                  ),
                ),
              ),

            // 예측 결과
            if (_result != null) ...[
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                color: _colorForLabel(_result!).withOpacity(0.1),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
                  child: Column(
                    children: [
                      const Text('예측 결과', style: TextStyle(fontSize: 14, color: Colors.grey)),
                      const SizedBox(height: 8),
                      Text(
                        _result!,
                        style: TextStyle(
                          fontSize: 56,
                          fontWeight: FontWeight.bold,
                          color: _colorForLabel(_result!),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),

              if (_probabilities != null)
                Card(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('클래스별 확률', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                        const SizedBox(height: 14),
                        ..._probabilities!.entries.map((e) {
                          final label = e.key;
                          final prob = (e.value as num).toDouble();
                          final color = _colorForLabel(label);
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 6),
                            child: Row(
                              children: [
                                SizedBox(
                                  width: 44,
                                  child: Text(label, style: TextStyle(fontWeight: FontWeight.bold, color: color)),
                                ),
                                Expanded(
                                  child: ClipRRect(
                                    borderRadius: BorderRadius.circular(6),
                                    child: LinearProgressIndicator(
                                      value: prob,
                                      minHeight: 14,
                                      backgroundColor: Colors.grey.shade200,
                                      valueColor: AlwaysStoppedAnimation<Color>(color),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Text('${(prob * 100).toStringAsFixed(1)}%', style: const TextStyle(fontWeight: FontWeight.w600)),
                              ],
                            ),
                          );
                        }),
                      ],
                    ),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}
