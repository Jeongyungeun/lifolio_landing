document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('opacity-100', 'translate-y-0');
                entry.target.classList.remove('opacity-0', 'translate-y-10');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach((el) => {
        el.classList.add('transition', 'duration-1000', 'ease-out', 'opacity-0', 'translate-y-10');
        observer.observe(el);
    });

    // 모달 관련 코드
    const modal = document.getElementById('emailModal');
    const modalContent = document.getElementById('modalContent');
    const closeModalBtn = document.getElementById('closeModal');
    const emailForm = document.getElementById('emailForm');
    const successMessage = document.getElementById('successMessage');
    const formContent = document.getElementById('formContent');

    // CTA 버튼들에 클릭 이벤트 추가
    const ctaButtons = document.querySelectorAll('.cta-button');
    ctaButtons.forEach(btn => {
        btn.addEventListener('click', openModal);
    });

    function openModal() {
        modal.classList.remove('hidden');
        setTimeout(() => {
            modal.classList.remove('opacity-0');
            modalContent.classList.remove('scale-95');
            modalContent.classList.add('scale-100');
        }, 10);
    }

    function closeModal() {
        modal.classList.add('opacity-0');
        modalContent.classList.remove('scale-100');
        modalContent.classList.add('scale-95');
        setTimeout(() => {
            modal.classList.add('hidden');
            // 폼 리셋
            if (formContent) formContent.classList.remove('hidden');
            if (successMessage) successMessage.classList.add('hidden');
            if (emailForm) emailForm.reset();
        }, 300);
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', closeModal);
    }

    // 모달 배경 클릭 시 닫기
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    }

    // 이메일 폼 제출
    if (emailForm) {
        emailForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('emailInput').value;
            const submitBtn = emailForm.querySelector('button[type="submit"]');

            // 버튼 비활성화
            submitBtn.disabled = true;
            submitBtn.textContent = '전송 중...';

            try {
                // Formspree로 이메일 전송
                const response = await fetch('https://formspree.io/f/xojjqpjp', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: email,
                        source: window.location.pathname
                    })
                });

                if (response.ok) {
                    // Google Ads 전환 추적
                    if (typeof gtag !== 'undefined') {
                        gtag('event', 'conversion', {
                            'send_to': 'AW-17872201033/submit_email',
                            'event_callback': function() {
                                console.log('Conversion tracked');
                            }
                        });
                    }

                    // 성공 메시지 표시
                    formContent.classList.add('hidden');
                    successMessage.classList.remove('hidden');

                    // 3초 후 모달 닫기
                    setTimeout(closeModal, 3000);
                } else {
                    throw new Error('전송 실패');
                }
            } catch (error) {
                alert('전송에 실패했습니다. 다시 시도해주세요.');
                submitBtn.disabled = false;
                submitBtn.textContent = '할인 쿠폰 받기';
            }
        });
    }

    // ESC 키로 모달 닫기
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });
});
